import os
import random
from functools import partial

import numpy as np
import tensorflow as tf
from absl import logging
from official.utils.misc import distribution_utils, keras_utils

import smart_compose.train.train_flow_helper
import smart_compose.train.train_model_helper
import smart_compose.utils.distributed_utils
from smart_compose.run_smart_compose import SmartComposeArg
from smart_compose.train import train_flow_helper
from smart_compose.train.metrics import get_metric_fn_lst
from smart_compose.train.optimization import clip_by_global_norm
from smart_compose.train.train_flow_helper import run_evaluation, train_step_fn, eval_step_fn, write_training_summary, export_best_model


def run_customized_training_loop(
        strategy,
        model_fn,
        loss_fn,
        out_dir,
        train_input_fn,
        eval_input_fn,
        test_input_fn,
        steps_per_stats,
        num_train_steps,
        steps_per_eval,
        run_eagerly,
        explicit_allreduce,
        pmetric_name,
        all_metric_fn_lst,
        keep_checkpoint_max,
        feature_type_2_name,
        optimizer_fn,
        bert_optimizer_fn,
        num_eval_steps,
        scale_loss=True,
        custom_callbacks=None,
        pre_allreduce_callbacks=None,
        post_allreduce_callbacks=None):
    """Run Smart Compose model training using low-level API

    Reference: official/nlp/bert/model_training_utils.py

    :param strategy: Distribution strategy on which to run low level training loop.
    :param model_fn: Function that returns a tuple (model, sub_model). Caller of this function should add optimizer to the `model` via calling
      `model.compile()` API or manually setting `model.optimizer` attribute.
    :param loss_fn: Function with signature func(labels, logits) and returns a loss tensor.
    :param scale_loss: Whether to divide the raw loss by number of replicas before gradients calculation.
    :param keep_checkpoint_max: Maximum checkpoints to keep.
    :param feature_type_2_name: Mapping from feature types to feature names.
    :param out_dir: Model directory used during training for restoring/saving model weights.
    :param train_input_fn: Function that returns a tf.data.Dataset used for training.
    :param eval_input_fn: Function that returns evaluation dataset
    :param test_input_fn: Function that returns test dataset
    :param num_train_steps: Total training steps.
    :param steps_per_eval: Number of steps to run per eval. At the end of each eval, model checkpoint will be saved and evaluation will be conducted
      if evaluation dataset is provided.
    :param steps_per_stats: Number of steps per graph-mode loop. In order to reduce communication in eager context, training logs are printed every
      steps_per_stats.
    :param optimizer_fn: Function to create optimizer for non-bert parameters
    :param bert_optimizer_fn: Function to create optimzier for bert parameters
    :param pmetric_name: Primary metric name. Model with best pmetric on evaluation dataset will be exported in pb format
    :param all_metric_fn_lst: A list of metrics functions that return a Keras Metric object to record evaluation result using evaluation dataset
    :param custom_callbacks: A list of Keras Callbacks objects to run during training. More specifically, `on_batch_begin()`, `on_batch_end()`,
      methods are invoked during training.
    :param run_eagerly: Whether to run model training in pure eager execution. This should be disable for TPUStrategy.
    :param explicit_allreduce: Whether to explicitly perform gradient allreduce, instead of relying on implicit allreduce in optimizer.apply_gradients().
      default is False. For now, if training using FP16 mixed precision, explicit allreduce will aggregate gradients in FP16 format. For TPU and
      GPU training using FP32, explicit allreduce will aggregate gradients in FP32 format.
    :param pre_allreduce_callbacks: A list of callback functions that takes gradients and model variables pairs as input, manipulate them, and returns
      a new gradients and model variables paris. The callback functions will be invoked in the list order and before gradients are allreduced.
      With mixed precision training, the pre_allreduce_allbacks will be applied on scaled_gradients. Default is no callbacks. Only used when
      explicit_allreduce=True.
    :param post_allreduce_callbacks: A list of callback functions that takes gradients and model variables pairs as input, manipulate them, and
      returns a new gradients and model variables paris. The callback functions will be invoked in the list order and right before gradients
      are applied to variables for updates. Default is no callbacks. Only used when explicit_allreduce=True.

    :return Trained model.
    """

    assert tf.executing_eagerly(), 'Outer training flow should be executed in eager mode'

    train_iterator = train_flow_helper.get_input_iterator(train_input_fn, strategy)
    with distribution_utils.get_strategy_scope(strategy):
        # To correctly place the model weights on accelerators, model and optimizer should be created in scope.
        model = model_fn()

        optimizer = optimizer_fn()
        bert_optimizer = bert_optimizer_fn()

        # Get metrics
        train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        all_metrics = [metric_fn() for metric_fn in all_metric_fn_lst] if all_metric_fn_lst else []
        pmetric = list(filter(lambda x: x.name == pmetric_name, all_metrics))[0]

        # Create summary writers
        summary_dir = train_flow_helper.create_summary_dir(strategy, out_dir)
        eval_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'eval'))
        train_summary_writer = train_flow_helper.create_train_summary_writer(os.path.join(summary_dir, 'train'), steps_per_stats)

        def train_step(iterator):
            """Performs a distributed training step """

            def _train_step(inputs):
                train_step_fn(inputs, model, loss_fn, scale_loss, strategy, explicit_allreduce, feature_type_2_name,
                              pre_allreduce_callbacks, post_allreduce_callbacks, optimizer, bert_optimizer, train_loss_metric)

            strategy.run(_train_step, args=(next(iterator),))

        def eval_step(iterator):
            """Calculates evaluation metrics on distributed devices."""

            def _eval_step(inputs):
                eval_step_fn(inputs, model, feature_type_2_name, all_metrics)

            strategy.run(_eval_step, args=(next(iterator),))

        # Use AutoGraph for efficiency if not in eager mode
        if not run_eagerly:
            train_step = tf.function(train_step)
            eval_step = tf.function(eval_step)

        # Manage checkpoints
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, bert_optimizer=bert_optimizer)
        manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=out_dir, max_to_keep=keep_checkpoint_max)
        if manager.latest_checkpoint:
            logging.info('Checkpoint file found and restoring')
            checkpoint.restore(manager.latest_checkpoint)

        current_step = optimizer.iterations.numpy()
        best_pmetric_val = None

        # Training loop starts here
        while current_step < num_train_steps:
            # Training loss/metric are taking average over steps inside micro training loop. We reset the their values before each round.
            train_loss_metric.reset_states()
            for metric in model.metrics:
                metric.reset_states()

            train_flow_helper.run_callbacks_on_batch_begin(current_step, custom_callbacks)
            steps = train_flow_helper.steps_to_run(current_step, steps_per_eval, steps_per_stats)

            for _ in range(steps):
                train_step(train_iterator)

            train_loss = train_flow_helper.float_metric_value(train_loss_metric)
            current_step += steps
            train_flow_helper.run_callbacks_on_batch_end(current_step - 1, {'loss': train_loss}, custom_callbacks)
            write_training_summary(train_summary_writer, train_loss, train_loss_metric, current_step)

            logging.info('Train Step: %d/%d  / loss = %s' % (current_step, num_train_steps, train_loss))

            # Saves model checkpoints and run validation steps
            if current_step % steps_per_eval == 0:
                # To avoid repeated model saving, we do not save after the last step of training.
                if current_step < num_train_steps:
                    train_flow_helper.save_checkpoint(strategy, current_step, manager)

                logging.info('Running evaluation after step: %s.', current_step)
                dev_iter = train_flow_helper.get_input_iterator(eval_input_fn, strategy, is_eval=True)
                run_evaluation(current_step, dev_iter, eval_summary_writer, all_metrics, eval_step, 'Validation', num_eval_steps)

                best_pmetric_val = export_best_model(strategy, pmetric, best_pmetric_val, model, out_dir)

                # Re-initialize evaluation metric
                for metric in all_metrics:
                    metric.reset_states()

        train_flow_helper.save_checkpoint(strategy, current_step, manager)

        logging.info('Running final evaluation after training is complete.')
        dev_iter = train_flow_helper.get_input_iterator(eval_input_fn, strategy, is_eval=True)
        run_evaluation(current_step, dev_iter, eval_summary_writer, all_metrics, eval_step, 'Validation', num_eval_steps)

        best_pmetric_val = export_best_model(strategy, pmetric, best_pmetric_val, model, out_dir)

        logging.info(f'Running evaluation on test set with best model on dev set ({pmetric_name} = {best_pmetric_val})')
        for metric in all_metrics:
            metric.reset_states()
        model.load_weights(train_flow_helper.get_export_model_variable_dir(out_dir)).expect_partial()  # TODO: remove expect partial
        test_iter = train_flow_helper.get_input_iterator(eval_input_fn, strategy, is_eval=True)
        run_evaluation(current_step, test_iter, eval_summary_writer, all_metrics, eval_step, 'Test', num_eval_steps)

        if not smart_compose.utils.distributed_utils.should_export_summary(strategy):
            tf.io.gfile.rmtree(summary_dir)

        return model


def train(strategy, hparams: SmartComposeArg):
    """ Main function for train/evaluate Smart Compose model

    :param strategy Distributed training strategy
    :param hparams HParams
    """
    random.seed(hparams.random_seed)
    np.random.seed(hparams.random_seed)
    tf.random.set_seed(hparams.random_seed)

    # Input functions
    train_input_fn = smart_compose.train.train_model_helper.get_input_fn_common(hparams.train_file, hparams.train_batch_size, tf.estimator.ModeKeys.TRAIN,
                                                                                hparams)
    eval_input_fn = smart_compose.train.train_model_helper.get_input_fn_common(hparams.dev_file, hparams.test_batch_size, tf.estimator.ModeKeys.EVAL, hparams)
    test_input_fn = smart_compose.train.train_model_helper.get_input_fn_common(hparams.test_file, hparams.test_batch_size, tf.estimator.ModeKeys.EVAL, hparams)

    # Callbacks for logging performance.
    custom_callbacks = [keras_utils.TimeHistory(batch_size=hparams.train_batch_size * strategy.num_replicas_in_sync,
                                                log_steps=hparams.steps_per_stats,
                                                logdir=hparams.out_dir)]

    # Loss, metrics and optimizers
    loss_fn = smart_compose.train.train_model_helper.get_loss_fn(hparams)
    metric_fn_lst = get_metric_fn_lst(hparams.all_metrics)
    optimizer_fn = smart_compose.train.train_model_helper.get_optimizer_fn(hparams)
    bert_optimizer_fn = smart_compose.train.train_model_helper.get_bert_optimizer_fn(hparams)
    model = run_customized_training_loop(model_fn=smart_compose.train.train_model_helper.get_model_fn(hparams),
                                         train_input_fn=train_input_fn,
                                         eval_input_fn=eval_input_fn,
                                         test_input_fn=test_input_fn,
                                         strategy=strategy,
                                         loss_fn=loss_fn,
                                         num_train_steps=hparams.num_train_steps,
                                         steps_per_stats=hparams.steps_per_stats,
                                         steps_per_eval=hparams.steps_per_eval,
                                         out_dir=hparams.out_dir,
                                         custom_callbacks=custom_callbacks,
                                         pre_allreduce_callbacks=[partial(clip_by_global_norm, clip_norm=hparams.max_gradient_norm)],
                                         all_metric_fn_lst=metric_fn_lst,
                                         explicit_allreduce=hparams.explicit_allreduce,
                                         run_eagerly=hparams.run_eagerly,
                                         pmetric_name=hparams.pmetric,
                                         feature_type_2_name=hparams.feature_type_2_name,
                                         keep_checkpoint_max=hparams.keep_checkpoint_max,
                                         optimizer_fn=optimizer_fn,
                                         bert_optimizer_fn=bert_optimizer_fn,
                                         num_eval_steps=hparams.num_eval_steps)

    train_flow_helper.print_model_summary(model)
    return model
