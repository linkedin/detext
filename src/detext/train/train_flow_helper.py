import os
import random
import tempfile

import numpy as np
import tensorflow as tf
from absl import logging

from detext.train.constant import Constant
from detext.train.optimization import process_grads_and_vars_using_explicit_allreduce, process_grads_and_vars_without_explicit_allreduce, \
    split_bert_grads_and_vars
from detext.train.train_model_helper import get_model_input, get_weight
from detext.utils.distributed_utils import should_export_summary, should_export_checkpoint
from detext.utils.parsing_utils import InputFtrType

_MIN_SUMMARY_STEPS = 10
_SCORES = 'score'


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def create_summary_dir(strategy, out_dir):
    # Create summary writers
    if should_export_summary(strategy):
        summary_dir = os.path.join(out_dir, 'summaries')
    else:
        # In multi worker training we need every worker to write summary, because variables can trigger synchronization on read and
        #   synchronization needs all workers to participate.
        summary_dir = tempfile.mkdtemp()
    return summary_dir


def run_callbacks_on_batch_begin(batch, custom_callbacks):
    """Runs custom callbacks at the start of every step."""
    if not custom_callbacks:
        return
    for callback in custom_callbacks:
        callback.on_batch_begin(batch)


def create_train_summary_writer(directory, steps_per_loop):
    if steps_per_loop < _MIN_SUMMARY_STEPS:
        return None
    # Only writes summary when the stats are collected sufficiently over enough steps
    return tf.summary.create_file_writer(directory)


def run_callbacks_on_batch_end(batch, logs, custom_callbacks):
    """Runs custom callbacks at the end of every step."""
    if not custom_callbacks:
        return
    for callback in custom_callbacks:
        callback.on_batch_end(batch, logs)


def get_input_iterator(input_fn, strategy, is_eval=False):
    """Returns distributed dataset iterator."""
    # When training with TPU pods, datasets need to be cloned across workers. Since Dataset instance cannot be cloned in eager mode, we instead
    #   pass callable that returns a dataset.
    if not callable(input_fn):
        raise ValueError('`input_fn` should be a closure that returns a dataset.')
    # To avoid data reshape issue caused by unevenly distributed samples across multiple devices,
    # we will not use the distributed dataset in the evaluation process. This way, each device will receive all the
    # samples instead.
    if is_eval:
        iterator = iter(input_fn(None))
    else:
        iterator = iter(strategy.distribute_datasets_from_function(input_fn))
    return iterator


def float_metric_value(metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)


def steps_to_run(current_step, steps_per_eval, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
        return steps_per_loop
    remainder_in_eval = current_step % steps_per_eval
    if remainder_in_eval != 0:
        return min(steps_per_eval - remainder_in_eval, steps_per_loop)
    else:
        return steps_per_loop


def save_checkpoint(strategy, step, manager: tf.train.CheckpointManager):
    """Saves model to with provided checkpoint prefix."""

    if should_export_checkpoint(strategy):
        saved_path = manager.save(step)
        logging.info('Saving model as TF checkpoint: %s', saved_path)
    else:
        # In multi worker training we need every worker to save checkpoint, because variables can trigger synchronization on read and synchronization needs
        #   all workers to participate. To avoid workers overriding each other we save to a temporary directory on non-chief workers.
        tmp_dir = tempfile.mkdtemp()
        manager.save(step)
        tf.io.gfile.rmtree(tmp_dir)


def print_model_summary(model: tf.keras.Model):
    """Prints model summary

    The summary includes two parts:
      1. model summary using tf.keras.Model.summary. This contains model parameters number in total and related input & output shape
      2. number of parameters related to representation layer, multi-layer perceptron and other parameters
    """
    model.summary(print_fn=logging.info)

    prefix2module = {'deep_match/rep_layer': {'count': 0}, 'deep_match/multi_layer_perceptron': {'count': 0}}
    others = {'count': 0}

    def get_dct(name):
        for prefix, dct in prefix2module.items():
            if name.startswith(prefix):
                return dct
        return others

    for variable in model.trainable_variables:
        name = variable.name
        dct = get_dct(name)
        dct[name] = variable.shape
        dct['count'] += tf.size(variable).numpy()

    for prefix, dct in prefix2module.items():
        logging.info(f'Number of parameters related to {prefix}: {dct["count"]}')
    logging.info(f'Number of other parameters: {others["count"]}')


def run_evaluation(current_training_step, test_iterator, eval_summary_writer, all_metrics, test_step, dataset, num_eval_steps):
    """Runs validation steps and aggregate metrics."""
    current_eval_step = 0
    while True:
        if num_eval_steps > 0:
            if current_eval_step >= num_eval_steps:
                break
            current_eval_step += 1
        try:
            test_step(test_iterator)
        except (StopIteration, tf.errors.OutOfRangeError):
            break

    with eval_summary_writer.as_default():
        for metric in all_metrics:
            if metric.name == 'confusion_matrix':
                # confusion_matrix is not a scalar.
                cm = metric.result().numpy()
                # We write the summary as image first
                # normalize the pixels
                cm_image = cm / tf.expand_dims(tf.reduce_sum(cm, axis=1), axis=-1)
                # invert pixels so that 0 -> white 1->black
                cm_image = 1 - cm_image
                # add 1st dim and channel dim to image for plotting
                cm_image = tf.expand_dims(tf.expand_dims(cm_image, axis=0), axis=-1)
                tf.summary.image(metric.name, cm_image, step=current_training_step)

                # also write the values of confusion_matrix (name: confusion_matrix_value) into summary for reference
                cm_str = '\n'.join(','.join('%0.4f' % x for x in y) for y in cm)
                tf.summary.text(metric.name + '_value', cm_str, step=current_training_step)
            else:
                metric_value = float_metric_value(metric)
                logging.info('Step: [%d] %s %s = %f', current_training_step, dataset, metric.name, metric_value)
                tf.summary.scalar(metric.name, metric_value, step=current_training_step)
        eval_summary_writer.flush()


def train_step_fn(inputs, model, loss_fn, scale_loss, strategy, explicit_allreduce, feature_type2name,
                  pre_allreduce_callbacks, post_allreduce_callbacks, optimizer, bert_optimizer,
                  task_ids, task_weights, train_loss_metric):
    """Training step"""

    inputs, labels = inputs
    with tf.GradientTape() as tape:
        scores = model.generate_training_scores(inputs=get_model_input(inputs, feature_type2name), training=True)
        loss = loss_fn(scores=scores, labels=labels[feature_type2name[InputFtrType.LABEL_COLUMN_NAME]],
                       weight=get_weight(inputs, labels, feature_type2name, task_ids, task_weights),
                       trainable_vars=model.trainable_variables)
        raw_loss = loss  # Raw loss is used for reporting in metrics/logs.
        if scale_loss:  # Scales down the loss for gradients to be invariant from replicas.
            loss = loss / strategy.num_replicas_in_sync

    experiment_aggregate_gradient = not explicit_allreduce
    if explicit_allreduce:
        grads_and_vars = process_grads_and_vars_using_explicit_allreduce(
            tape, optimizer, loss, model.trainable_variables, pre_allreduce_callbacks, post_allreduce_callbacks)
    else:
        grads_and_vars = process_grads_and_vars_without_explicit_allreduce(tape, optimizer, loss, model.trainable_variables, post_allreduce_callbacks)

    bert_grads_and_vars, non_bert_grads_and_vars = split_bert_grads_and_vars(grads_and_vars)
    optimizer.apply_gradients(non_bert_grads_and_vars, experimental_aggregate_gradients=experiment_aggregate_gradient)
    if bert_grads_and_vars:
        bert_optimizer.apply_gradients(bert_grads_and_vars, experimental_aggregate_gradients=experiment_aggregate_gradient)

    # For reporting, the metric takes the mean of losses.
    train_loss_metric.update_state(raw_loss)


def predict_step_fn(inputs, model, feature_type2name):
    """Returns predicted scores for given inputs"""
    return model.generate_training_scores(inputs=get_model_input(inputs, feature_type2name), training=False)


def eval_step_fn(inputs, model, feature_type2name, all_metrics):
    """Replicated metric calculation."""
    inputs, labels = inputs
    scores = predict_step_fn(inputs, model, feature_type2name)
    for metric in all_metrics:
        metric.update_state(labels[feature_type2name[InputFtrType.LABEL_COLUMN_NAME]], scores)


def predict_with_additional_info(dataset: tf.data.Dataset, model: tf.keras.Model, feature_type2name):
    """Predicts output for given dataset"""

    @tf.function
    def _predict_with_additional_info(inputs, label):
        """Predicts scores with additional info (uid, weight, label) """
        uid_ftr_name = feature_type2name.get(InputFtrType.UID_COLUMN_NAME, Constant()._DEFAULT_UID_FTR_NAME)
        weight_ftr_name = feature_type2name.get(InputFtrType.WEIGHT_COLUMN_NAME, Constant()._DEFAULT_WEIGHT_FTR_NAME)
        label_ftr_name = feature_type2name[InputFtrType.LABEL_COLUMN_NAME]
        return {
            _SCORES: predict_step_fn(inputs, model, feature_type2name),
            uid_ftr_name: label[uid_ftr_name],
            weight_ftr_name: label[weight_ftr_name],
            label_ftr_name: label[label_ftr_name]
        }

    for inputs, labels in dataset:
        yield _predict_with_additional_info(inputs, labels)


def write_training_summary(train_summary_writer, train_loss, train_loss_metric, current_step):
    """Writes training summary"""
    if train_summary_writer:
        with train_summary_writer.as_default():
            tf.summary.scalar(train_loss_metric.name, train_loss, step=current_step)
            train_summary_writer.flush()


def get_export_dir(out_dir):
    """Returns directory of exported best model"""
    return os.path.join(out_dir, 'export')


def get_export_model_variable_dir(out_dir):
    """Returns directory of exported best model weights"""
    return os.path.join(get_export_dir(out_dir), 'variables', 'variables')


def export_best_model(strategy, pmetric, best_pmetric_val, model: tf.keras.Model, out_dir):
    """Exports best model if current primary metric value is better than the best metric value"""
    pmetric_name = pmetric.name
    pmetric_val = float_metric_value(pmetric)
    if best_pmetric_val is not None:
        logging.info(f'Existing best primary metric ({pmetric_name}): {best_pmetric_val}')
    logging.info(f'Current primary metric ({pmetric_name}): {pmetric_val}')
    if best_pmetric_val is None or pmetric_val > best_pmetric_val:
        logging.info(f'Exporting model with new best primary metric ({pmetric.name}): {pmetric_val}')
        if should_export_checkpoint(strategy):
            model.save(get_export_dir(out_dir))
        else:
            # In multi worker training we need every worker to save models, because variables can trigger synchronization on read and synchronization needs
            #   all workers to participate. To avoid workers overriding each other we save to a temporary directory on non-chief workers.
            tmp_dir = tempfile.mkdtemp()
            model.save(get_export_dir(out_dir))
            tf.io.gfile.rmtree(tmp_dir)
        return pmetric_val
    return best_pmetric_val
