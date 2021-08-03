import tensorflow as tf

from detext.utils.parsing_utils import InputFtrType


class FeatureGrouper(tf.keras.layers.Layer):
    """Feature grouper

    Features of the same type (e.g. dense numeric features) will be concatenated into one vector
    """

    def __init__(self):
        super(FeatureGrouper, self).__init__()

    def process_single_ftr(self, inputs, ftr_type, func):
        """Apply function on the input wrt the given feature type """
        if ftr_type in inputs:
            inputs[ftr_type] = func(inputs[ftr_type])
        return inputs

    def process_list_ftr(self, inputs, ftr_type, func):
        """Applies function on every element of the input list wrt the given feature types

        The function is applied on each feature tensor (corresponding to one feature name)
        """
        if ftr_type in inputs:
            result = []
            for tensor in inputs[ftr_type]:
                result.append(func(tensor))
            inputs[ftr_type] = result
        return inputs

    def call(self, inputs, *args, **kwargs):
        """Processes input features """
        inputs = inputs.copy()
        # Concatenate features that supports a list of inputs
        # E.g., users may have two arrays of dense features, one named "demographics", one named "professional". Since DeText treat them as dense features,
        # we concatenate them into one array
        self.process_single_ftr(inputs, InputFtrType.DENSE_FTRS_COLUMN_NAMES, concat_on_last_axis_dense)

        return inputs


def concat_on_last_axis_sparse(tensor_list):
    """Concatenates list of sparse tensors on the last axis"""
    return tf.sparse.concat(sp_inputs=tensor_list, axis=-1)


def concat_on_last_axis_dense(tensor_list):
    """Concatenates list of dense tensors on the last axis"""
    return tf.concat(tensor_list, axis=-1)
