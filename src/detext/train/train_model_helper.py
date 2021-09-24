import inspect
from collections import OrderedDict
from functools import partial

import tensorflow as tf

from detext.layers.id_embed_layer import IdEmbedLayer
from detext.layers.representation_layer import RepresentationLayer
from detext.layers.representation_layer import _FTR_EXT_NAME_TO_ENCODER
from detext.train.constant import Constant
from detext.train.data_fn import input_fn_tfrecord
from detext.train.loss import compute_loss
from detext.train.model import create_detext_model
from detext.train.optimization import create_optimizer
from detext.utils.parsing_utils import HParams
from detext.utils.parsing_utils import InputFtrType, iterate_items_with_list_val


def get_model_input(inputs, feature_type2name: dict):
    """ Returns DeText model inputs in the format of OrderedDict

    Unrelated features are filtered out
    """
    all_ftr_name = list()
    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
        if ftr_type in {InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.LABEL_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME}:
            continue
        all_ftr_name += ftr_name_lst

    return OrderedDict(sorted([(ftr_name, inputs[ftr_name]) for ftr_name in all_ftr_name]))


def get_weight(features, labels, feature_type2name, task_ids, task_weights):
    """ Returns the weights adjusted with task_weights

    :param features: dict containing the features in data
    :param labels: dict containing the labels in data
    :param feature_type2name: dict containing mapping from feature names to feature types
    :param task_ids Task ids
    :param task_weights Task weights
    """
    # For multitask training
    weight_ftr_name = feature_type2name.get(InputFtrType.WEIGHT_COLUMN_NAME, Constant()._DEFAULT_WEIGHT_FTR_NAME)
    weight = labels[weight_ftr_name]

    # Update the weight with each task's weight such that weight per document = weight * task_weight
    if task_ids is not None:
        task_id_field = features[feature_type2name[InputFtrType.TASK_ID_COLUMN_NAME]]  # shape=[batch_size,]
        task_ids = task_ids  # e.g. [0, 1, 2]
        task_weights = task_weights  # e.g. [0.1, 0.3, 0.6]
        # Expand task_id_field with shape [batch_size, num_tasks]
        expanded_task_id_field = tf.transpose(tf.broadcast_to(task_id_field, [len(task_ids), tf.shape(task_id_field)[0]]))
        task_mask = tf.cast(tf.equal(expanded_task_id_field, task_ids), dtype=tf.float32)
        weight *= tf.reduce_sum(task_mask * task_weights, 1)  # shape=[batch_size,]

    return weight


def get_input_fn_common(pattern, batch_size, mode, hparams):
    """ Returns the common input function used in DeText training and evaluation"""
    return _get_input_fn_common(pattern, batch_size, mode,
                                **_get_func_param_from_hparams(_get_input_fn_common, hparams, ('pattern', 'batch_size', 'mode')))


def _get_input_fn_common(pattern, batch_size, mode, task_type, feature_type2name, feature_name2num: dict):
    """ Returns the common input function used in DeText training and evaluation"""
    return lambda ctx: input_fn_tfrecord(
        input_pattern=pattern, batch_size=batch_size, mode=mode, task_type=task_type,
        feature_type2name=feature_type2name,
        feature_name2num=feature_name2num,
        input_pipeline_context=ctx
    )


def _get_vocab_layer_for_id_ftr_param(hparams):
    """ Extracts required parameters for VocabLayer for id features by function signature from hparams """
    return HParams(CLS='',
                   SEP='',
                   PAD=hparams.PAD_FOR_ID_FTR,
                   UNK=hparams.UNK_FOR_ID_FTR,
                   vocab_file=hparams.vocab_file_for_id_ftr)


def _get_vocab_layer_param(hparams):
    """ Extracts required parameters for VocabLayer by function signature from hparams """
    from detext.layers.vocab_layer import VocabLayerFromPath
    param_dct = _get_func_param_from_hparams(VocabLayerFromPath.__init__, hparams)
    return HParams(**param_dct)


def _get_embedding_layer_for_id_ftr_param(hparams):
    """ Extracts required parameters for EmbeddingLayer for id features by function signature from hparams """
    return HParams(vocab_hub_url=hparams.vocab_hub_url_for_id_ftr,
                   we_file=hparams.we_file_for_id_ftr,
                   we_trainable=hparams.we_trainable_for_id_ftr,
                   num_units=hparams.num_units_for_id_ftr,
                   vocab_layer_param=_get_vocab_layer_for_id_ftr_param(hparams))


def _get_embedding_layer(hparams):
    """ Extracts required parameters for EmbeddingLayer by function signature from hparams """
    from detext.layers.embedding_layer import EmbeddingLayer
    param_dct = _get_func_param_from_hparams(EmbeddingLayer.__init__, hparams, exclude_lst=('self', 'vocab_layer_param', 'name_prefix'))
    return HParams(**param_dct, vocab_layer_param=_get_vocab_layer_param(hparams))


def _get_id_encoder_param(hparams):
    """ Extracts required parameters for IdEmbedLayer by function signature from hparams """
    param_dct = _get_func_param_from_hparams(IdEmbedLayer.__init__, hparams, exclude_lst=('self', 'embedding_layer_param'))
    return HParams(**param_dct, embedding_layer_param=_get_embedding_layer_for_id_ftr_param(hparams))


def _get_text_encoder_param(hparams):
    """Extracts required parameters for text encoder layer by function signature from hparams """
    param_dct = _get_func_param_from_hparams(_FTR_EXT_NAME_TO_ENCODER[hparams.ftr_ext].__init__, hparams,
                                             exclude_lst=('self', 'embedding_layer_param', 'kwargs'))
    return HParams(**param_dct, embedding_layer_param=_get_embedding_layer(hparams))


def _get_rep_layer_param(hparams):
    """ Extracts required parameters for RepLayer by function signature from hparams """
    text_encoder_param = _get_text_encoder_param(hparams)
    id_encoder_param = _get_id_encoder_param(hparams)

    rep_layer_param_dct = {'text_encoder_param': text_encoder_param,
                           'id_encoder_param': id_encoder_param,
                           **_get_func_param_from_hparams(RepresentationLayer.__init__, hparams, ('self', 'id_encoder_param', 'text_encoder_param'))}

    return HParams(**rep_layer_param_dct)


def _get_func_param_from_hparams(func, hparams: HParams, exclude_lst=('self', 'args', 'kwargs')) -> dict:
    """ Extracts required parameters by the function signature from hparams. Used by DeText dev only

    This function saves the trouble in specifying the long list of params required by DeText layers, models, and functions. Note that names of the given
        function should be attributes of hparams (i.e., if there's a param in func named "p1", hparams.p1 must exist)
    :param func Target function
    :param hparams Parameter holder
    :param exclude_lst List of parameters to exclude. There are two cases to use customized setting for this parameter --
        1. parameters that are not directly accessible from hparams. Such as
            1) rep_layer_param in DeepMatch
            2) text_encoder_param and id_encoder_param in RepLayer
        2. parameters to be exposed when creating a partial function. Such as
            1) get_loss_fn. For this function, we want to emphasize the parameters taken by the partial functions, which are
                'scores', 'labels', 'weight', 'trainable_vars', as specified in exclude_lst
    """
    param_dct = dict()
    param_lst = inspect.signature(func)
    for param in param_lst.parameters:
        if param in exclude_lst:
            continue
        param_dct[param] = getattr(hparams, param)

    return param_dct


def get_model_fn(hparams):
    """Returns a lambda function that creates a DeText model from hparams"""
    return lambda: create_detext_model(
        rep_layer_param=_get_rep_layer_param(hparams),
        **_get_func_param_from_hparams(create_detext_model, hparams, ('rep_layer_param',))
    )


def get_optimizer_fn(hparams):
    """ Returns function that creates an optimizer for non bert parameters"""
    return lambda: create_optimizer(init_lr=hparams.learning_rate,
                                    num_train_steps=hparams.num_train_steps,
                                    num_warmup_steps=hparams.num_warmup_steps,
                                    optimizer=hparams.optimizer,
                                    use_lr_schedule=hparams.use_lr_schedule,
                                    use_bias_correction_for_adamw=hparams.use_bias_correction_for_adamw)


def get_bert_optimizer_fn(hparams):
    """ Returns function that creates an optimizer for bert parameters """
    return lambda: create_optimizer(init_lr=hparams.lr_bert,
                                    num_train_steps=hparams.num_train_steps,
                                    num_warmup_steps=hparams.num_warmup_steps,
                                    optimizer=hparams.optimizer,
                                    use_lr_schedule=hparams.use_lr_schedule,
                                    use_bias_correction_for_adamw=hparams.use_bias_correction_for_adamw)


def get_loss_fn(hparams):
    """ Returns a partial function that returns the loss function"""
    loss_fn = partial(compute_loss, **_get_func_param_from_hparams(compute_loss, hparams, exclude_lst=('scores', 'labels', 'weight', 'trainable_vars')))
    return loss_fn


def load_model_with_ckpt(hparams, path):
    """Returns loaded model with given checkpoint path"""
    model = get_model_fn(hparams)()
    model.load_weights(path)
    return model
