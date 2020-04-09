import tensorflow as tf
import tensorflow_ranking as tfr
import os
from os.path import join as path_join

from detext.model.deep_match import DeepMatch
from detext.model.lambdarank import LambdaRank
from detext.train import metrics
from detext.train import optimization, train_helper
from detext.train.data_fn import input_fn
from detext.train.loss import compute_softmax_loss, compute_sigmoid_cross_entropy_loss, \
    compute_regularization_penalty
from detext.utils import vocab_utils, executor_utils
from detext.utils.best_checkpoint_copier import BestCheckpointCopier


def train(hparams):
    """
    Main function for train/evaluate DeText ranking model
    :param hparams: hparams
    :return:
    """
    eval_log_file = None
    if hparams.use_horovod:
        import horovod.tensorflow as hvd
        eval_log_file = path_join(hparams.out_dir, 'eval_log.txt')

    # Set model export config for evaluator or primary worker of horovod
    if not hparams.use_horovod or (hparams.use_horovod and hvd.rank() == 0):
        best_model_name = 'best_' + hparams.pmetric
        # Exporter to save best (in terms of pmetric) checkpoint in the folder [best_model_name],
        # and export to savedmodel for prediction.
        best_checkpoint_exporter = BestCheckpointCopier(
            name=best_model_name,
            serving_input_receiver_fn=lambda: serving_input_fn(hparams),
            checkpoints_to_keep=1,  # keeping the best checkpoint
            exports_to_keep=1,  # keeping the best savedmodel
            pmetric='metric/{}'.format(hparams.pmetric),
            compare_fn=lambda x, y: x.score > y.score,  # larger metric better
            sort_reverse=True,
            eval_log_file=eval_log_file)

    # Handle single gpu or async distributed training case
    if not hparams.use_horovod:
        config = tf.estimator.RunConfig(
            save_summary_steps=hparams.steps_per_stats,
            save_checkpoints_steps=hparams.steps_per_eval,
            log_step_count_steps=hparams.steps_per_stats,
            keep_checkpoint_max=hparams.keep_checkpoint_max,
            train_distribute=tf.contrib.distribute.ParameterServerStrategy()
            )
        # TO DO:
        # In the future we will support both sync training and async training on mlearn.
        # Now the sync training can run on kubernetes cluster using hvd lib.
        # 1. async training.
        # train_distribute=tf.contrib.distribute.ParameterServerStrategy()  #async strategy.
        # 2. sync strategy
        # train_distribute=tf.distribute.experimental.MultiWorkerMirroredStrategy() #sync strategy,
        # and remember to remove ps in tony params when use sync strategy,
        # Also note that MultiWorkerMirroredStrategy doesn't support bert-adam optimizer, but can use adam optimizer

        # Create TrainSpec for model training
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(input_pattern=hparams.train_file,
                                      metadata_path=hparams.metadata_path,
                                      batch_size=hparams.train_batch_size,
                                      mode=tf.estimator.ModeKeys.TRAIN,
                                      vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                      vocab_table_for_id_ftr=vocab_utils.read_tf_vocab(hparams.vocab_file_for_id_ftr,
                                                                                       hparams.UNK_FOR_ID_FTR),
                                      feature_names=hparams.feature_names,
                                      CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                      PAD_FOR_ID_FTR=hparams.PAD_FOR_ID_FTR,
                                      max_len=hparams.max_len,
                                      min_len=hparams.min_len,
                                      cnn_filter_window_size=max(
                                          hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
                                      ),
            max_steps=hparams.num_train_steps)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(input_pattern=hparams.dev_file,
                                      metadata_path=hparams.metadata_path,
                                      batch_size=hparams.test_batch_size,
                                      mode=tf.estimator.ModeKeys.EVAL,
                                      vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                      vocab_table_for_id_ftr=vocab_utils.read_tf_vocab(hparams.vocab_file_for_id_ftr,
                                                                                       hparams.UNK_FOR_ID_FTR),
                                      feature_names=hparams.feature_names,
                                      CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                      PAD_FOR_ID_FTR=hparams.PAD_FOR_ID_FTR,
                                      max_len=hparams.max_len,
                                      min_len=hparams.min_len,
                                      cnn_filter_window_size=max(
                                          hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
                                      ),
            exporters=[best_checkpoint_exporter],
            steps=None,
            # Set throttle_secs to 10 instead of 0 to avoid warning to spam logs
            throttle_secs=10,
            start_delay_secs=10)

        # Build the estimator
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=hparams.out_dir, params=hparams, config=config)

        # Training and evaluation with dev set
        tf.estimator.train_and_evaluate(
            estimator=estimator,
            train_spec=train_spec,
            eval_spec=eval_spec
        )

    # Handle sync distributed training case via use_horovod
    else:
        # Adjust config for horovod
        session_config = tf.ConfigProto()
        session_config.allow_soft_placement = True
        # Pin each worker to a GPU
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())

        config = tf.estimator.RunConfig(
            save_summary_steps=hparams.steps_per_stats,
            save_checkpoints_steps=hparams.steps_per_eval,
            log_step_count_steps=hparams.steps_per_stats,
            keep_checkpoint_max=hparams.keep_checkpoint_max,
            session_config=session_config)

        model_dir = hparams.out_dir if hvd.rank() == 0 else None
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=hparams, config=config)

        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
        # rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights or
        # restored from a checkpoint.
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

        # tf.estimator.train_and_evaluate doesn't work with horovod: https://github.com/horovod/horovod/issues/182
        # One workaround is to pass evaluation listeners to the primary worker's estimator.train() to evaluate when
        # a new checkpoint is saved, but this won't work for sync distributed training as workers are expected to
        # train at close speed otherwise there'll be syncup timeout issue. See issue: https://github.com/horovod/horovod/issues/403
        # For now let all workers to do eval.

        # Extract global step from checkpoint filename
        current_step = 0
        if tf.gfile.Exists(path_join(hparams.out_dir, "checkpoint")):
            ckpt = tf.train.get_checkpoint_state(hparams.out_dir)
            current_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[-1])

        # There is dataset issue with wrapping train and evaluate in a for loop, in every iteration the data_fn
        # will construct a new dataset object and call make_initializable_iterator to create an iterator from
        # the beginning of the dataset. In other words, there is no guarentee all data will be iterated.
        # To alleviate this issue, recommend to:
        # - use larger steps_per_eval
        # - make each data file smaller (e.g. 128M), to make them same probability to be the first elements of the iterator
        while current_step < hparams.num_train_steps:
            steps_per_eval = min(hparams.steps_per_eval, hparams.num_train_steps - current_step)
            current_step += hparams.steps_per_eval
            estimator.train(
                input_fn=lambda: input_fn(input_pattern=hparams.train_file,
                                          metadata_path=hparams.metadata_path,
                                          batch_size=hparams.train_batch_size,
                                          mode=tf.estimator.ModeKeys.TRAIN,
                                          vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                          vocab_table_for_id_ftr=vocab_utils.read_tf_vocab(
                                              hparams.vocab_file_for_id_ftr,
                                              hparams.UNK_FOR_ID_FTR),
                                          feature_names=hparams.feature_names,
                                          CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                          PAD_FOR_ID_FTR=hparams.PAD_FOR_ID_FTR,
                                          max_len=hparams.max_len,
                                          min_len=hparams.min_len,
                                          cnn_filter_window_size=max(
                                              hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0,
                                          hvd_info=hparams.hvd_info),
                steps=steps_per_eval,
                hooks=[bcast_hook])
            # Evaluate every hparams.steps_per_eval steps
            eval_result = estimator.evaluate(
                input_fn=lambda: input_fn(input_pattern=hparams.dev_file,
                                          metadata_path=hparams.metadata_path,
                                          batch_size=hparams.test_batch_size,
                                          mode=tf.estimator.ModeKeys.EVAL,
                                          vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                          vocab_table_for_id_ftr=vocab_utils.read_tf_vocab(
                                              hparams.vocab_file_for_id_ftr,
                                              hparams.UNK_FOR_ID_FTR),
                                          feature_names=hparams.feature_names,
                                          CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                          PAD_FOR_ID_FTR=hparams.PAD_FOR_ID_FTR,
                                          max_len=hparams.max_len,
                                          min_len=hparams.min_len,
                                          cnn_filter_window_size=max(
                                              hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0)
            )
            tf.logging.info("Eval On Dev Set: {}".format(eval_result))
            if hvd.rank() == 0:
                best_checkpoint_exporter.export(
                    estimator=estimator,
                    export_path=path_join(
                        tf.compat.as_str_any(estimator.model_dir),
                        tf.compat.as_str_any("export/{}".format(best_checkpoint_exporter.name))),
                    checkpoint_path=estimator.latest_checkpoint(),
                    eval_result=eval_result,
                    is_the_final_export=False)

    # Evaluation with test set: create an estimator with the best_checkpoint_dir to load the best model
    task_type = executor_utils.get_executor_task_type()
    do_evaluate = task_type == executor_utils.EVALUATOR or task_type == executor_utils.LOCAL_MODE
    if (not hparams.use_horovod and do_evaluate) or (hparams.use_horovod and hvd.rank() == 0):
        best_checkpoint_dir = path_join(hparams.out_dir, best_model_name)
        estimator_savedmodel = tf.estimator.Estimator(model_fn=model_fn, model_dir=best_checkpoint_dir,
                                                      params=hparams, config=config)
        result = estimator_savedmodel.evaluate(
            input_fn=lambda: input_fn(input_pattern=hparams.test_file,
                                      metadata_path=hparams.metadata_path,
                                      batch_size=hparams.test_batch_size,
                                      mode=tf.estimator.ModeKeys.EVAL,
                                      vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
                                      vocab_table_for_id_ftr=vocab_utils.read_tf_vocab(
                                          hparams.vocab_file_for_id_ftr,
                                          hparams.UNK_FOR_ID_FTR),
                                      feature_names=hparams.feature_names,
                                      CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
                                      PAD_FOR_ID_FTR=hparams.PAD_FOR_ID_FTR,
                                      max_len=hparams.max_len,
                                      min_len=hparams.min_len,
                                      cnn_filter_window_size=max(
                                          hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0)
        )
        print("\n\n***** Eval results of best model on test data: *****")
        for key in sorted(result.keys()):
            print("%s = %s" % (key, str(result[key])))


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

    doc_id_feature_names = [df for df in hparams.feature_names if df.startswith('docId_')]
    usr_id_feature_names = [df for df in hparams.feature_names if df.startswith('usrId_')]
    doc_id_fields, doc_id_placeholders = train_helper.get_doc_id_fields(hparams)
    usr_id_fields, usr_id_placeholders = train_helper.get_usr_id_fields(hparams)

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
        elif fname.startswith('docId_'):
            if fname not in features:
                for fi in range(hparams.num_doc_id_fields):
                    features[doc_id_feature_names[fi]] = doc_id_fields[fi]
                    feature_placeholders[doc_id_feature_names[fi]] = doc_id_placeholders[fi]
        elif fname.startswith('usrId_'):
            if fname not in features:
                for fi in range(hparams.num_usr_id_fields):
                    features[usr_id_feature_names[fi]] = usr_id_fields[fi]
                    feature_placeholders[usr_id_feature_names[fi]] = usr_id_placeholders[fi]
        elif fname == 'wide_ftrs':
            feature_placeholders[fname] = wide_ftr_placeholder
            features[fname] = wide_ftrs
        elif fname == 'wide_ftrs_sp_val':
            feature_placeholders[fname] = wide_ftr_sp_val_placeholder
            features[fname] = wide_ftrs_sp_val
        elif fname == 'wide_ftrs_sp_idx':
            feature_placeholders[fname] = wide_ftr_sp_idx_placeholder
            features[fname] = wide_ftrs_sp_idx
        elif fname == 'label' or fname == 'weight':
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

    weight = features.get('weight', None)
    wide_ftrs = features.get('wide_ftrs', None)

    wide_ftrs_sp_idx = features.get('wide_ftrs_sp_idx', None)
    wide_ftrs_sp_val = features.get('wide_ftrs_sp_val', None)

    doc_fields = [features[ftr_name] for ftr_name in features if ftr_name.startswith('doc_')]
    if len(doc_fields) == 0:
        doc_fields = None

    usr_fields = [features[ftr_name] for ftr_name in features if ftr_name.startswith('usr_')]
    if len(usr_fields) == 0:
        usr_fields = None

    doc_id_fields = [features[ftr_name] for ftr_name in features if ftr_name.startswith('docId_')]
    if len(doc_id_fields) == 0:
        doc_id_fields = None

    usr_id_fields = [features[ftr_name] for ftr_name in features if ftr_name.startswith('usrId_')]
    if len(usr_id_fields) == 0:
        usr_id_fields = None

    label_field = labels['label'] if mode != tf.estimator.ModeKeys.PREDICT else None

    group_size_field = features['group_size'] if mode != tf.estimator.ModeKeys.PREDICT else None

    # build graph
    model = DeepMatch(query=query_field,
                      wide_ftrs=wide_ftrs,
                      doc_fields=doc_fields,
                      usr_fields=usr_fields,
                      doc_id_fields=doc_id_fields,
                      usr_id_fields=usr_id_fields,
                      hparams=params,
                      mode=mode,
                      wide_ftrs_sp_idx=wide_ftrs_sp_idx,
                      wide_ftrs_sp_val=wide_ftrs_sp_val)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = compute_loss(params, model.scores, label_field, group_size_field, weight)
        train_op, _, _ = optimization.create_optimizer(params, loss)
        global_step = tf.train.get_global_step()
        train_tensors_log = {'loss': loss, 'global_step': global_step}
        logging_hook = tf.train.LoggingTensorHook(train_tensors_log, every_n_iter=10)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook])

    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = compute_loss(params, model.scores, label_field, group_size_field, weight)
        eval_metric_ops = {}
        for metric_name in params.all_metrics:
            metric_op_name = 'metric/{}'.format(metric_name)
            topk = int(metric_name.split('@')[1]) if '@' in metric_name else 10  # Default topk
            if metric_name.startswith('ndcg'):
                eval_metric_ops[metric_op_name] = metrics.compute_ndcg_tfr(
                    model.scores, label_field, features, topk)
            elif metric_name.startswith('mrr'):
                eval_metric_ops[metric_op_name] = metrics.compute_mrr_tfr(
                    model.scores, label_field, features)
            elif metric_name.startswith('precision'):
                eval_metric_ops[metric_op_name] = metrics.compute_precision_tfr(
                    model.scores, label_field, features, topk)
            elif metric_name.startswith('traditional_ndcg'):
                eval_metric_ops[metric_op_name] = metrics.compute_ndcg(
                    model.scores, label_field, group_size_field, topk)
            elif metric_name.startswith('li_mrr'):
                eval_metric_ops[metric_op_name] = metrics.compute_mrr(
                    model.scores, labels['label'], features['group_size'], topk)
            elif metric_name == 'auc':
                eval_metric_ops[metric_op_name] = metrics.compute_auc(
                    model.scores, label_field
                )
            elif metric_name == 'accuracy':
                eval_metric_ops[metric_op_name] = metrics.compute_accuracy(model.scores, label_field)
            elif metric_name == 'confusion_matrix':
                eval_metric_ops[metric_op_name] = metrics.compute_confusion_matrix(model.scores, label_field,
                                                                                   params.num_classes)
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
        # multiclass classification: export the probabilities across classes by applying softmax
        if params.num_classes > 1:
            predictions['multiclass_probabilities'] = tf.nn.softmax(model.scores)

        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        # Provide an estimator spec for `ModeKeys.PREDICT` mode.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)
    else:
        raise ValueError("Only support mode as TRAIN/EVAL/PREDICT")


def compute_loss(hparams, scores, labels, group_size, weight):
    """ Computes ranking/classification loss with regularization """
    if weight is None:
        raise ValueError("weight should not be None")
    return compute_rank_clf_loss(hparams, scores, labels, group_size, weight) + tf.reduce_mean(
        weight) * compute_regularization_penalty(hparams)


def compute_rank_clf_loss(hparams, scores, labels, group_size, weight):
    """
    Compute ranking/classification loss
    Note that the tfr loss is slightly different than our implementation: the tfr loss is sum over all loss and
    devided by number of queries; our implementation is sum over all loss and devided by the number of larger than
    0 labels.
    """
    # Classification loss
    if hparams.num_classes > 1:
        labels = tf.cast(labels, tf.int32)
        labels = tf.squeeze(labels, -1)  # Last dimension is max_group_size, which should be 1
        return tf.losses.sparse_softmax_cross_entropy(logits=scores, labels=labels, weights=weight)

    # Expand weight to [batch size, 1] so that in inhouse ranking loss it can be multiplied with loss which is
    #   [batch_size, max_group_size]
    expanded_weight = tf.expand_dims(weight, axis=-1)

    # Ranking losses
    # tf-ranking loss
    if hparams.use_tfr_loss:
        weight_name = "weight"
        loss_fn = tfr.losses.make_loss_fn(hparams.tfr_loss_fn, lambda_weight=hparams.tfr_lambda_weights,
                                          weights_feature_name=weight_name)
        loss = loss_fn(labels, scores, {weight_name: expanded_weight})
        return loss

    # our own implementation
    if hparams.ltr_loss_fn == 'pairwise':
        lambdarank = LambdaRank()
        pairwise_loss, pairwise_mask = lambdarank(scores, labels, group_size)
        loss = tf.reduce_sum(tf.reduce_sum(pairwise_loss, axis=[1, 2]) * expanded_weight) / tf.reduce_sum(pairwise_mask)
    elif hparams.ltr_loss_fn == 'softmax':
        loss = compute_softmax_loss(scores, labels, group_size) * expanded_weight
        is_positive_label = tf.cast(tf.greater(labels, 0), dtype=tf.float32)
        loss = tf.div_no_nan(tf.reduce_sum(loss), tf.reduce_sum(is_positive_label))
    elif hparams.ltr_loss_fn == 'pointwise':
        loss = compute_sigmoid_cross_entropy_loss(scores, labels, group_size) * expanded_weight
        loss = tf.reduce_mean(loss)
    else:
        raise ValueError('Currently only support pointwise/pairwise/softmax/softmax_cls.')
    return loss
