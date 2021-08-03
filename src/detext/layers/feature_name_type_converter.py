from functools import partial

import tensorflow as tf

from detext.utils.parsing_utils import InputFtrType, TaskType, InternalFtrType

DEEP_FTR_TYPES = {
    InputFtrType.QUERY_COLUMN_NAME,
    InputFtrType.DOC_TEXT_COLUMN_NAMES,
    InputFtrType.DOC_ID_COLUMN_NAMES,
    InputFtrType.USER_TEXT_COLUMN_NAMES,
    InputFtrType.USER_ID_COLUMN_NAMES
}

WIDE_FTR_TYPES = {
    InputFtrType.DENSE_FTRS_COLUMN_NAMES,
    InputFtrType.SPARSE_FTRS_COLUMN_NAMES
}

MULTITASK_FTR_TYPES = {
    InputFtrType.TASK_ID_COLUMN_NAME
}

DOC_FTR_TYPES = {
    InputFtrType.DOC_TEXT_COLUMN_NAMES,
    InputFtrType.DOC_ID_COLUMN_NAMES,
    InputFtrType.DENSE_FTRS_COLUMN_NAMES,
    InputFtrType.SPARSE_FTRS_COLUMN_NAMES
}


def get_wide_features_inputs_ranking(features, feature_type2name):
    """ Returns wide ranking features by parsing features from data input_fn

    :param features: dict containing features in data
    :param feature_type2name: dict mapping feature types to feature names
    """
    result = {}
    for ftr_type, ftr_name_lst in feature_type2name.items():
        if ftr_type not in WIDE_FTR_TYPES:
            continue
        assert isinstance(ftr_name_lst, list), "Wide features must accept a list of feature names"
        result[ftr_type] = [features[ftr_name] for ftr_name in ftr_name_lst]

    return {k: v for k, v in result.items() if v is not None}


def get_deep_features_inputs_ranking(features, feature_type2name):
    """ Returns deep ranking features by parsing features from data input_fn

    :param features: dict containing the features in data
    :param feature_type2name: dict mapping feature types to feature names
    """
    result = {}
    for ftr_type, ftr_name_lst in feature_type2name.items():
        if ftr_type not in DEEP_FTR_TYPES:
            continue
        if isinstance(ftr_name_lst, list):
            # Feature type accepting a list of feature names. E.g., doc_text, doc_id, user_text, user_id, etc.
            result[ftr_type] = [features[ftr_name] for ftr_name in ftr_name_lst]
        else:
            # Feature type accepting only one feature name. E.g., query, label, weight
            result[ftr_type] = features[ftr_name_lst]

    return {k: v for k, v in result.items() if v is not None}


def get_multitask_features_inputs(features, feature_type2name):
    """ Returns deep ranking features by parsing features from data input_fn

    :param features: dict containing the features in data
    :param feature_type2name: dict mapping feature types to feature names
    """
    result = {}
    for ftr_type, ftr_name_lst in feature_type2name.items():
        if ftr_type not in MULTITASK_FTR_TYPES:
            continue
        # Feature type accepting only one feature name
        result[ftr_type] = features[ftr_name_lst]

    return {k: v for k, v in result.items() if v is not None}


def get_wide_features_inputs_classification(features, feature_type2name):
    """ Returns wide classification features by parsing features from data input_fn

    :param features: dict containing features in data
    :param feature_type2name: dict mapping feature types to feature names
    """
    result = {}
    for ftr_type, ftr_name_lst in feature_type2name.items():
        if ftr_type not in WIDE_FTR_TYPES:
            continue
        result[ftr_type] = []
        assert isinstance(ftr_name_lst, list), "Wide features must accept a list of feature names"

        for ftr_name in ftr_name_lst:
            feat = features[ftr_name]
            if ftr_type in DOC_FTR_TYPES:  # Expand the list dimension (axis=1) for classification
                if ftr_type == InputFtrType.SPARSE_FTRS_COLUMN_NAMES:
                    feat = tf.sparse.expand_dims(feat, axis=1, name=ftr_name)
                else:
                    feat = tf.expand_dims(feat, axis=1, name=ftr_name)
            result[ftr_type].append(feat)

    return {k: v for k, v in result.items() if v is not None}


def get_deep_features_inputs_classification(features, feature_type2name):
    """Returns deep classification features by parsing features from data input_fn

    :param features: dict containing features in data
    :param feature_type2name: dict mapping feature types to feature names
    """
    result = {}
    for ftr_type, ftr_name_lst in feature_type2name.items():
        if ftr_type not in DEEP_FTR_TYPES:
            continue
        result[ftr_type] = []
        if isinstance(ftr_name_lst, list):
            # Feature type accepting a list of feature names. E.g., doc_text, doc_id, user_text, user_id, etc.
            for ftr_name in ftr_name_lst:
                feat = features[ftr_name]
                if ftr_type in DOC_FTR_TYPES:  # Expand the list dimension (axis=1) for classification
                    feat = tf.expand_dims(feat, axis=1, name=ftr_name)
                result[ftr_type].append(feat)
        else:
            # Feature type accepting only one feature name. E.g., query, label, weight
            feat = features[ftr_name_lst]
            result[ftr_type] = feat

    return {k: v for k, v in result.items() if v is not None}


TASK_TYPE_TO_GET_WIDE_FEATURES_FN = {
    TaskType.RANKING: get_wide_features_inputs_ranking,
    TaskType.CLASSIFICATION: get_wide_features_inputs_classification,
    TaskType.BINARY_CLASSIFICATION: get_wide_features_inputs_classification
}

TASK_TYPE_TO_GET_DEEP_FEATURES_FN = {
    TaskType.RANKING: get_deep_features_inputs_ranking,
    TaskType.CLASSIFICATION: get_deep_features_inputs_classification,
    TaskType.BINARY_CLASSIFICATION: get_deep_features_inputs_classification
}


class FeatureNameTypeConverter(tf.keras.layers.Layer):
    """Converter that converts feature_name->tensor mapping to feature_type->tensor mapping

    By using feature type (static and known internally in DeText) instead of feature names (dynamic), the Detext layer design becomes independent of user
        feature naming and free from the trouble of keeping notes about feature_type->feature_names
    """

    def __init__(self, task_type, feature_type2name, **kwargs):
        super(FeatureNameTypeConverter, self).__init__(**kwargs)
        self._task_type = task_type
        self._feature_type2name = feature_type2name

        self.converters = {
            InternalFtrType.WIDE_FTR_BAG: partial(TASK_TYPE_TO_GET_WIDE_FEATURES_FN[task_type], feature_type2name=feature_type2name),
            InternalFtrType.DEEP_FTR_BAG: partial(TASK_TYPE_TO_GET_DEEP_FEATURES_FN[task_type], feature_type2name=feature_type2name),
            InternalFtrType.MULTITASK_FTR_BAG: partial(get_multitask_features_inputs, feature_type2name=feature_type2name)
        }

    def call(self, inputs, **kwargs):
        return {k: self.converters[k](v) for k, v in inputs.items()}
