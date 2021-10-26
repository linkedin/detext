import inspect
from collections import OrderedDict
from functools import partial

from smart_compose.args import SmartComposeArg
from smart_compose.train.data_fn import input_fn_tfrecord
from smart_compose.train.losses import compute_loss
from smart_compose.train.model import create_smart_compose_model
from smart_compose.train.optimization import create_optimizer
from smart_compose.utils.parsing_utils import HParams

LR_BERT = 0.001


def get_model_input(inputs, feature_type_2_name: dict):
    """ Returns Smart Compose model inputs in the format of OrderedDict

    Unrelated features are filtered out
    """
    return OrderedDict(sorted([(ftr_name, inputs[ftr_name]) for ftr_type, ftr_name in feature_type_2_name.items()]))


def get_input_fn_common(pattern, batch_size, mode, hparams: SmartComposeArg):
    """ Returns the common input function used in Smart Compose training and evaluation"""
    return _get_input_fn_common(pattern, batch_size, mode,
                                **_get_func_param_from_hparams(_get_input_fn_common, hparams, ('pattern', 'batch_size', 'mode')))


def _get_input_fn_common(pattern, batch_size, mode, feature_type_2_name):
    """ Returns the common input function used in Smart Compose training and evaluation"""
    return lambda ctx: input_fn_tfrecord(
        input_pattern=pattern, batch_size=batch_size, mode=mode,
        feature_type_2_name=feature_type_2_name,
        input_pipeline_context=ctx
    )


def _get_vocab_layer_param(hparams: SmartComposeArg):
    """ Extracts required parameters for VocabLayer by function signature from hparams """
    from smart_compose.layers.vocab_layer import VocabLayerFromPath
    param_dct = _get_func_param_from_hparams(VocabLayerFromPath.__init__, hparams)
    return HParams(**param_dct)


def _get_embedding_layer_param(hparams: SmartComposeArg):
    """ Extracts required parameters for EmbeddingLayer by function signature from hparams """
    from smart_compose.layers.embedding_layer import EmbeddingLayer
    param_dct = _get_func_param_from_hparams(EmbeddingLayer.__init__, hparams, exclude_lst=('self', 'vocab_layer_param', 'name_prefix'))
    return HParams(**param_dct, vocab_layer_param=_get_vocab_layer_param(hparams))


def _get_func_param_from_hparams(func, hparams: HParams, exclude_lst=('self', 'args', 'kwargs')) -> dict:
    """ Extracts required parameters by the function signature from hparams. Used by Smart Compose dev only

    This function saves the trouble in specifying the long list of params required by Smart Compose layers, models, and functions. Note that names of the given
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


def get_model_fn(hparams: SmartComposeArg):
    """Returns a lambda function that creates a Smart Compose model from hparams"""
    return lambda: create_smart_compose_model(_get_embedding_layer_param(hparams),
                                              **_get_func_param_from_hparams(create_smart_compose_model,
                                                                             hparams,
                                                                             exclude_lst=('self', 'embedding_layer_param')))


def get_optimizer_fn(hparams):
    """ Returns function that creates an optimizer for non bert parameters"""
    return lambda: create_optimizer(hparams.learning_rate, hparams.num_train_steps, hparams.num_warmup_steps, hparams.optimizer,
                                    hparams.use_bias_correction_for_adamw)


def get_bert_optimizer_fn(hparams):
    """ Returns function that creates an optimizer for bert parameters """
    return lambda: create_optimizer(LR_BERT, hparams.num_train_steps, hparams.num_warmup_steps, hparams.optimizer,
                                    hparams.use_bias_correction_for_adamw)


def get_loss_fn(hparams: SmartComposeArg):
    """ Returns a partial function that returns the loss function"""
    loss_fn = partial(compute_loss, l1=hparams.l1, l2=hparams.l2)
    return loss_fn


def load_model_with_ckpt(hparams, path):
    """Returns loaded model with given checkpoint path"""
    model = get_model_fn(hparams)()
    model.load_weights(path)
    return model
