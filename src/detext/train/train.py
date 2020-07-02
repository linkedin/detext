import tensorflow as tf
import tensorflow_ranking as tfr
from os.path import join as path_join

from detext.model.deep_match import DeepMatch
from detext.model.lambdarank import LambdaRank
from detext.train import metrics
from detext.train import optimization, train_helper
from detext.train.loss import compute_softmax_loss, compute_sigmoid_cross_entropy_loss, \
    compute_regularization_penalty
from detext.utils import vocab_utils, executor_utils
from detext.utils.best_checkpoint_copier import BestCheckpointCopier


def train(hparams, input_fn):
    """
    Main function for train/evaluate DeText ranking model
    :param hparams: hparams
    :param input_fn: input function to create train/eval specs
    :return:
    """
    if hparams.use_horovod is True:
        import horovod.tensorflow as hvd
    else:
        hvd = None
    train_strategy = tf.contrib.distribute.ParameterServerStrategy()
    estimator = get_estimator(hparams, strategy=train_strategy)

    # Set model export config for evaluator or primary worker of horovod
    exporter_list = None
    if not hvd or hvd.rank() == 0:
        best_model_name = 'best_' + hparams.pmetric
        # Exporter to save best (in terms of pmetric) checkpoint in the folder [best_model_name],
        # and export to savedmodel for prediction.
        best_checkpoint_exporter = BestCheckpointCopier(
            name=best_model_name,
            serving_input_receiver_fn=lambda: serving_input_fn(hparams),
            checkpoints_to_keep=1,  # keeping the best checkpoint
            exports_to_keep=1,  # keeping the best savedmodel
            pmetric=f'metric/{hparams.pmetric}',
            compare_fn=lambda x, y: x.score > y.score,  # larger metric better
            sort_reverse=True)
        exporter_list = [best_checkpoint_exporter]

    # Handle sync distributed training case via use_horovod

        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
        # rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights or
        # restored from a checkpoint.
    bcast_hook = [hvd.BroadcastGlobalVariablesHook(0)] if hvd else None

    def input_fn_common(pattern, batch_size=hparams.test_batch_size, mode=tf.estimator.ModeKeys.EVAL, hvd_info=None):
        return lambda: input_fn(
            input_pattern=pattern, metadata_path=hparams.metadata_path, batch_size=batch_size, mode=mode,
            vocab_table=vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK), hvd_info=hvd_info,
            vocab_table_for_id_ftr=vocab_utils.read_tf_vocab(hparams.vocab_file_for_id_ftr, hparams.UNK_FOR_ID_FTR),
            feature_names=hparams.feature_names, CLS=hparams.CLS, SEP=hparams.SEP, PAD=hparams.PAD,
            PAD_FOR_ID_FTR=hparams.PAD_FOR_ID_FTR, max_len=hparams.max_len, min_len=hparams.min_len,
            cnn_filter_window_size=max(hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0)

    # Create TrainSpec for model training
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn_common(hparams.train_file,
                                 batch_size=hparams.train_batch_size,
                                 mode=tf.estimator.ModeKeys.TRAIN,
                                 # Add horovod information if applicable
                                 hvd_info=hparams.hvd_info if hvd else None),
        hooks=bcast_hook,
        max_steps=hparams.num_train_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn_common(hparams.dev_file),
        exporters=exporter_list,
        steps=None,
        # Set throttle to 0 to start evaluation right away.
        # (Note: throttle_secs has to be 0 for horovod:
        # https://github.com/horovod/horovod/issues/182#issuecomment-533897757)
        throttle_secs=0,
        start_delay_secs=10)

    # Training and evaluation with dev set
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec)
    print("***** Training finished. *****")

    # Evaluation with test set: create an estimator with the best_checkpoint_dir to load the best model
    task_type = executor_utils.get_executor_task_type()
    do_evaluate = task_type == executor_utils.EVALUATOR or task_type == executor_utils.LOCAL_MODE
    if (not hvd and do_evaluate) or (hvd and hvd.rank() == 0):
        best_checkpoint_dir = path_join(hparams.out_dir, best_model_name)
        estimator_savedmodel = get_estimator(hparams, strategy=train_strategy, best_checkpoint=best_checkpoint_dir)
        result = estimator_savedmodel.evaluate(input_fn=input_fn_common(hparams.test_file))
        print("\n***** Evaluation on test set with best exported model: *****")
        for key in sorted(result.keys()):
            print("%s = %s" % (key, str(result[key])))


def get_estimator(hparams, strategy, best_checkpoint=None):
    config_kwargs = {
        'save_summary_steps': hparams.steps_per_stats,
        'save_checkpoints_steps': hparams.steps_per_eval,
        'log_step_count_steps': hparams.steps_per_stats,
        'keep_checkpoint_max': hparams.keep_checkpoint_max
    }
    # Handle sync distributed training case via use_horovod
    if hparams.use_horovod:
        import horovod.tensorflow as hvd
        # Adjust config for horovod
        session_config = tf.ConfigProto()
        session_config.allow_soft_placement = True
        # Pin each worker to a GPU
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())
        config_kwargs['session_config'] = session_config

        model_dir = best_checkpoint if best_checkpoint is not None else hparams.out_dir if hvd.rank() == 0 else None
    # Handle single gpu or async distributed training case
    else:
        # TODO:
        # In the future we will support both sync training and async training on TOnY.
        # Now the sync training can run on kubernetes cluster using hvd lib.
        # 1. async training.
        # train_distribute=tf.contrib.distribute.ParameterServerStrategy()  #async strategy.
        # 2. sync strategy
        # train_distribute=tf.distribute.experimental.MultiWorkerMirroredStrategy() #sync strategy,
        # and remember to remove ps in tony params when use sync strategy,
        # Also note that MultiWorkerMirroredStrategy doesn't support bert-adam optimizer, but can use adam optimizer
        config_kwargs['train_distribute'] = strategy
        model_dir = best_checkpoint if best_checkpoint is not None else hparams.out_dir

    config = tf.estimator.RunConfig(**config_kwargs)
    # Build the estimator
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=hparams, config=config)


def serving_input_fn(hparams):
    """
    Creates a serving input function used by inference in model export.
    :param hparams: hparams
    :return: A ServingInputReceiver function that defines the inference requests and prepares for the model.
    """

    # Define placeholders and features
    doc_feature_names = [df for df in hparams.feature_names if df.startswith('doc_')]
    usr_feature_names = [df for df in hparams.feature_names if df.startswith('usr_')]
    doc_fields, doc_placeholders = train_helper.get_doc_fields(hparams)
    usr_fields, usr_placeholders = train_helper.get_usr_fields(hparams)

    doc_id_feature_names = [df for df in hparams.feature_names if df.startswith('docId_')]
    usr_id_feature_names = [df for df in hparams.feature_names if df.startswith('usrId_')]
    doc_id_fields, doc_id_placeholders = train_helper.get_doc_id_fields(hparams)
    usr_id_fields, usr_id_placeholders = train_helper.get_usr_id_fields(hparams)

    query, query_placeholder = train_helper.get_query(hparams)
    wide_ftr_placeholder, wide_ftrs = train_helper.create_placeholder_for_ftrs(
        "wide_ftr_placeholder", [None, hparams.num_wide], tf.float32, 'wide_ftrs', hparams.feature_names)
    wide_ftr_sp_idx_placeholder, wide_ftrs_sp_idx = train_helper.create_placeholder_for_ftrs(
        "wide_ftr_sp_idx_placeholder", [None, None], tf.int32, 'wide_ftrs_sp_idx', hparams.feature_names)
    wide_ftr_sp_val_placeholder, wide_ftrs_sp_val = train_helper.create_placeholder_for_ftrs(
        "wide_ftr_sp_val_placeholder", [None, None], tf.float32, 'wide_ftrs_sp_val', hparams.feature_names)
    uid_placeholder, uid = train_helper.create_placeholder_for_ftrs(
        "uid_placeholder", [], tf.int64, 'uid', hparams.feature_names, tf.constant([-1], dtype=tf.int64))
    weight_placeholder, weight = train_helper.create_placeholder_for_ftrs(
        "weight_placeholder", [], tf.float32, 'weight', hparams.feature_names, tf.constant([1.0], dtype=tf.float32))
    label_placeholder, label = train_helper.create_placeholder_for_ftrs(
        "label_placeholder", [None], tf.float32, 'label', hparams.feature_names)
    task_id_placeholder, task_id = train_helper.create_placeholder_for_ftrs(
        "task_id_placeholder", [], tf.int64, 'task_id', hparams.feature_names)
    # Placeholder tensors
    # Default uid as feature for detext integration, will be -1 by default if not present in data
    feature_placeholders = {'uid': uid_placeholder, 'weight': weight_placeholder, 'label': label_placeholder}
    # Features to feed into model (creating model_fn)
    features = {'uid': uid, 'weight': weight, 'label': label}
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
        elif fname == 'task_id':
            feature_placeholders[fname] = task_id_placeholder
            features[fname] = task_id
        elif fname == 'label' or fname == 'weight' or fname == 'uid':
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

    uid = features.get('uid', None)

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
    labels_passthrough = features['label']

    group_size_field = features['group_size'] if mode != tf.estimator.ModeKeys.PREDICT else None

    # For multitask training
    task_id_field = features.get('task_id', None)  # shape=[batch_size,]

    # Update the weight with each task's weight such that weight per document = weight * task_weight
    if params.task_ids is not None:
        task_ids = params.task_ids  # e.g. [0, 1, 2]
        task_weights = params.task_weights  # e.g. [0.1, 0.3, 0.6]
        # Expand task_id_field with shape [batch_size, num_tasks]
        expanded_task_id_field = tf.transpose(tf.broadcast_to(task_id_field, [len(task_ids), tf.shape(task_id_field)[0]]))
        task_mask = tf.cast(tf.equal(expanded_task_id_field, task_ids), dtype=tf.float32)
        weight *= tf.reduce_sum(task_mask * task_weights, 1)  # shape=[batch_size,]

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
                      wide_ftrs_sp_val=wide_ftrs_sp_val,
                      task_id_field=task_id_field)

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
                metric = metrics.compute_ndcg_tfr(model.scores, label_field, features, topk)
            elif metric_name.startswith('mrr'):
                metric = metrics.compute_mrr_tfr(model.scores, label_field, features)
            elif metric_name.startswith('precision'):
                metric = metrics.compute_precision_tfr(model.scores, label_field, features, topk)
            elif metric_name.startswith('traditional_ndcg'):
                metric = metrics.compute_ndcg(model.scores, label_field, group_size_field, topk)
            elif metric_name.startswith('li_mrr'):
                metric = metrics.compute_mrr(model.scores, labels['label'], features['group_size'], topk)
            elif metric_name == 'auc':
                metric = metrics.compute_auc(model.scores, label_field)
            elif metric_name == 'accuracy':
                metric = metrics.compute_accuracy(model.scores, label_field)
            elif metric_name == 'confusion_matrix':
                metric = metrics.compute_confusion_matrix(model.scores, label_field, params.num_classes)
            else:
                raise ValueError(f"Unsupported metrics: {metric_name}")
            eval_metric_ops[metric_op_name] = metric
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        # Prediction field for scoring models
        predictions = {
            'uid': uid,
            'scores': model.original_scores,
            'weight': weight,
            'label': labels_passthrough
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
