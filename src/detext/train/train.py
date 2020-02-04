from os.path import join as path_join

import tensorflow as tf
import tensorflow_ranking as tfr

from detext.model import metrics
from detext.model.deep_match import DeepMatch
from detext.model.lambdarank import LambdaRank
from detext.model.softmax_loss import compute_softmax_loss, compute_sigmoid_cross_entropy_loss
from detext.train import optimization, train_helper
from detext.train.data_fn import input_fn
from detext.utils import vocab_utils


def train(hparams):
    """
    Main function for train/evaluate DeText ranking model
    :param hparams: hparams
    :return:
    """
    eval_log_file = path_join(hparams.out_dir, 'eval_log.txt')
    config = tf.estimator.RunConfig(
        save_summary_steps=hparams.steps_per_stats,
        save_checkpoints_steps=hparams.steps_per_eval,
        log_step_count_steps=hparams.steps_per_stats,
        keep_checkpoint_max=hparams.keep_checkpoint_max)

    # Create TrainSpec for model training
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(input_pattern=hparams.train_file,
                                  batch_size=hparams.train_batch_size,
                                  mode=tf.estimator.ModeKeys.TRAIN,
                                  vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                  feature_names=hparams.feature_names,
                                  CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                  max_len=hparams.max_len,
                                  min_len=hparams.min_len,
                                  cnn_filter_window_size=max(
                                      hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
                                  ),
        max_steps=hparams.num_train_steps)

    model_exporter = tf.estimator.BestExporter(
        name='best_' + hparams.pmetric,
        # the export folder name
        serving_input_receiver_fn=lambda: serving_input_fn(hparams),
        event_file_pattern='eval/*.tfevents.*',
        compare_fn=lambda best_eval_result, current_eval_result:
        exporter_metric_compare_fn(best_eval_result, current_eval_result, 'metric/{}'.format(hparams.pmetric),
                                   eval_log_file),
        exports_to_keep=1)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(input_pattern=hparams.dev_file,
                                  batch_size=hparams.test_batch_size,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                  feature_names=hparams.feature_names,
                                  CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                  max_len=hparams.max_len,
                                  min_len=hparams.min_len,
                                  cnn_filter_window_size=max(
                                      hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
                                  ),
        exporters=[model_exporter],
        steps=None,
        # Set throttle_secs to 10 instead of 0 to avoid warning to spam logs
        throttle_secs=10)

    # Build the estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=hparams.out_dir, params=hparams, config=config)

    # Training and evaluation with dev set
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    # Evaluation with test set and write to file
    result = estimator.evaluate(
        input_fn=lambda: input_fn(input_pattern=hparams.test_file,
                                  batch_size=hparams.test_batch_size,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                  feature_names=hparams.feature_names,
                                  CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                  max_len=hparams.max_len,
                                  min_len=hparams.min_len,
                                  cnn_filter_window_size=max(
                                      hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
                                  ),
        steps=None)

    print("***** Eval results *****")
    print("Evaluation results on test set:")
    for key in sorted(result.keys()):
        print("{} = {}".format(key, result[key]))


def exporter_metric_compare_fn(best_eval_result, current_eval_result, metric_key, out_file):
    """
    A compare function to be used by tf.estimator.BestExporter for checking export criteria.
    If the current eval metric is larger than best eval metric, the function returns true, for performing export.
    :param best_eval_result: Best eval metrics.
    :param current_eval_result: Current eval metrics.
    :param metric_key: Which metric to use for determining whether to perform model export.
    :param out_file: File path to evaluation results logging file
    :return: True is current eval metric is better than previous best, else False.
    """

    # Helper function that prints messages using tf logging info and output the message to fout
    def print_message(fout, s):
        tf.compat.v1.logging.info(s)
        fout.write(s + '\n')

    # Log the metrics info.
    if not tf.io.gfile.exists(out_file):
        with tf.io.gfile.GFile(out_file, 'w') as fout:
            fout.write("***** Evaluation on Dev Set *****\n")

    # Print metrics and step info
    with tf.io.gfile.GFile(out_file, 'a') as fout:
        print_message(fout, "## Step {}".format(current_eval_result.get('global_step', -1)))
        print_message(fout, "## Best eval metric ({}) value:    {:.6f}".format(metric_key, best_eval_result[metric_key]))
        print_message(fout, "## Current eval metric ({}) value: {:.6f}".format(metric_key, current_eval_result[metric_key]))
        for metric in current_eval_result:
            if metric != metric_key and metric != 'global_step':
                print_message(fout, "##    {} : {:.6f}".format(metric, current_eval_result[metric]))

    # Check if metric_key is valid.
    if not best_eval_result or metric_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no %s is found in it.' % (metric_key))

    if not current_eval_result or metric_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no %s is found in it.' % (metric_key))

    # Add a signature when we have a better model. This helps us keep track of the best model on dev set
    is_better = best_eval_result[metric_key] < current_eval_result[metric_key]
    if is_better:
        print_message(fout, "## Better metric achieved!")

    return is_better


def serving_input_fn(hparams):
    """
    Creates a serving input function used by inference in model export.
    :param hparams: hparams
    :return: A ServingInputReceiver function that defines the inference requests and prepares for the model.
    """

    # Define placeholders and features
    doc_feature_names = [df for df in hparams.feature_names if df.startswith('doc_')]
    usr_feature_names = [df for df in hparams.feature_names if df.startswith('usr_')]
    doc_fields, doc_placeholders = train_helper.get_doc_fields(hparams, hparams.regex_replace_pattern)
    usr_fields, usr_placeholders = train_helper.get_usr_fields(hparams, hparams.regex_replace_pattern)
    query, query_placeholder = train_helper.get_query(hparams, hparams.regex_replace_pattern)
    wide_ftr_placeholder, wide_ftrs = train_helper.create_placeholder_for_ftrs(
        "wide_ftr_placeholder", [None, hparams.num_wide], tf.float32, 'wide_ftrs', hparams.feature_names)
    wide_ftr_sp_idx_placeholder, wide_ftrs_sp_idx = train_helper.create_placeholder_for_ftrs(
        "wide_ftr_sp_idx_placeholder", [None, None], tf.int32, 'wide_ftrs_sp_idx', hparams.feature_names)
    wide_ftr_sp_val_placeholder, wide_ftrs_sp_val = train_helper.create_placeholder_for_ftrs(
        "wide_ftr_sp_val_placeholder", [None, None], tf.float32, 'wide_ftrs_sp_val', hparams.feature_names)

    # Placeholder tensors
    feature_placeholders = {}
    # Features to feed into model (creating model_fn)
    features = {}
    # Add objects into pleaceholders and features
    for fname in hparams.feature_names:
        if fname == 'query':
            feature_placeholders[fname] = query_placeholder
            features[fname] = query
        elif fname.startswith('doc_'):
            if fname not in features:
                for fi in range(hparams.num_doc_fields):
                    features[doc_feature_names[fi]] = doc_fields[fi]
                    feature_placeholders[doc_feature_names[fi]] = doc_placeholders[fi]
        elif fname.startswith('usr_'):
            if fname not in features:
                for fi in range(hparams.num_usr_fields):
                    features[usr_feature_names[fi]] = usr_fields[fi]
                    feature_placeholders[usr_feature_names[fi]] = usr_placeholders[fi]
        elif fname == 'wide_ftrs':
            feature_placeholders[fname] = wide_ftr_placeholder
            features[fname] = wide_ftrs
        elif fname == 'wide_ftrs_sp_val':
            feature_placeholders[fname] = wide_ftr_sp_val_placeholder
            features[fname] = wide_ftrs_sp_val
        elif fname == 'wide_ftrs_sp_idx':
            feature_placeholders[fname] = wide_ftr_sp_idx_placeholder
            features[fname] = wide_ftrs_sp_idx
        elif fname == 'label':
            continue
        else:
            raise ValueError("Unsupported feature_to_add argument: %s" % fname)

    return tf.estimator.export.ServingInputReceiver(
        features, feature_placeholders)


def model_fn(features, labels, mode, params):
    """
    Defines the model_fn to feed in to estimator
    :param features: dict containing the features in data
    :param labels: dict containing labels in data
    :param mode: running mode, in TRAIN/EVAL/PREDICT
    :param params: hparams used
    :return: tf.estimator.EstimatorSpec
    """
    query_field = features.get('query', None)

    wide_ftrs = features.get('wide_ftrs', None)

    wide_ftrs_sp_idx = features.get('wide_ftrs_sp_idx', None)
    wide_ftrs_sp_val = features.get('wide_ftrs_sp_val', None)

    doc_fields = [features[ftr_name] for ftr_name in features if ftr_name.startswith('doc_')]
    if len(doc_fields) == 0:
        doc_fields = None
    else:
        assert len(doc_fields) == params.num_doc_fields, "num_doc_fields must be equal to number of document fields"

    usr_fields = [features[ftr_name] for ftr_name in features if ftr_name.startswith('usr_')]
    if len(usr_fields) == 0:
        usr_fields = None
    else:
        assert len(usr_fields) == params.num_usr_fields, "num_usr_fields must be equal to number of user fields"

    label_field = labels['label'] if mode != tf.estimator.ModeKeys.PREDICT else None

    group_size_field = features['group_size'] if mode != tf.estimator.ModeKeys.PREDICT else None

    # build graph
    model = DeepMatch(query=query_field,
                      wide_ftrs=wide_ftrs,
                      doc_fields=doc_fields,
                      usr_fields=usr_fields,
                      hparams=params,
                      mode=mode,
                      wide_ftrs_sp_idx=wide_ftrs_sp_idx,
                      wide_ftrs_sp_val=wide_ftrs_sp_val)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = compute_loss(params, model.scores, label_field, group_size_field)
        train_op, _, _ = optimization.create_optimizer(params, loss)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = compute_loss(params, model.scores, label_field, group_size_field)
        eval_metric_ops = {}
        for metric_name in params.all_metrics:
            metric_op_name = 'metric/{}'.format(metric_name)
            if metric_name.startswith('ndcg@'):
                topk = int(metric_name.split('@')[1])
                eval_metric_ops[metric_op_name] = metrics.compute_ndcg_tfr(
                    model.scores, label_field, features, topk)
            elif metric_name.startswith('mrr'):
                eval_metric_ops[metric_op_name] = metrics.compute_mrr_tfr(
                    model.scores, label_field, features)
            elif metric_name.startswith('precision@'):
                topk = int(metric_name.split('@')[1])
                eval_metric_ops[metric_op_name] = metrics.compute_precision_tfr(
                    model.scores, label_field, features, topk)
            elif metric_name.startswith('traditional_ndcg@'):
                topk = int(metric_name.split('@')[1])
                eval_metric_ops[metric_op_name] = metrics.compute_ndcg(
                    model.scores, label_field, group_size_field, topk)
            elif metric_name == 'auc':
                eval_metric_ops[metric_op_name] = metrics.compute_auc(
                    model.scores, label_field
                )
            else:
                raise ValueError("Unsupported metrics: %s" % (metric_name))

        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        # Prediction field for scoring models
        predictions = {
            'scores': model.scores
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        # Provide an estimator spec for `ModeKeys.PREDICT` mode.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)
    else:
        raise ValueError("Only support mode as TRAIN/EVAL/PREDICT")


def compute_loss(hparams, scores, labels, group_size):
    """
    Compute loss
    Note that the tfr loss is slightly different than our implementation: the tfr loss is sum over all loss and
    devided by number of queries; our implementation is sum over all loss and devided by the number of larger than
    0 labels.
    """
    # tf-ranking loss
    if hparams.use_tfr_loss:
        loss_fn = tfr.losses.make_loss_fn(hparams.tfr_loss_fn, lambda_weight=hparams.tfr_lambda_weights)
        loss = loss_fn(labels, scores, None)
        return loss

    def apply_lambdarank(self):
        """
        Apply lambda rank.
        """
        self.lambdarank = LambdaRank(self._hparams.lambda_metric)
        self.pairwise_loss, self.pairwise_mask = self.lambdarank(self.scores, self.labels, self._group_size)
        loss = tf.reduce_sum(self.pairwise_loss) / tf.reduce_sum(self.pairwise_mask)
        return loss

    # our own implementation
    if hparams.ltr_loss_fn == 'pairwise':
        loss = apply_lambdarank()
    elif hparams.ltr_loss_fn == 'softmax':
        loss = compute_softmax_loss(scores, labels, group_size)
        num_non_zeros = tf.cast(tf.greater(labels, 0), dtype=tf.float32)
        loss = tf.reduce_sum(loss) / tf.reduce_sum(num_non_zeros)
    elif hparams.ltr_loss_fn == 'pointwise':
        loss = compute_sigmoid_cross_entropy_loss(scores, labels, group_size)
        loss = tf.reduce_mean(loss)
    else:
        raise ValueError('Currently only support pointwise/pairwise/softmax.')
    return loss
