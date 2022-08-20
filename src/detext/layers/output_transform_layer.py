import tensorflow as tf

from detext.utils.parsing_utils import TaskType, OutputFtrType


def _ranking_output_transform(inputs):
    """
    Transforms the outputs for DeText ranking task.
    :param inputs: Tensor with shape [batch_size, list_size, 1].
    :return: final output for ranking, with shape [batch_size, list_size]
    """
    # shape: [batch_size, list_size]
    outputs = tf.squeeze(inputs, axis=-1)
    return {OutputFtrType.DETEXT_RANKING_SCORES: outputs}


def _classification_output_transform(inputs):
    """
    Transforms the outputs for DeText classification task.
    :param inputs: Tensor with shape [batch_size, 1, num_classes].
    :return: final output for classification, with shape [batch_size, num_classes]
    """
    # shape: [batch_size, num_classes]
    inputs = tf.squeeze(inputs, axis=-2)
    # Return logits, softmax, and label predictions for classification
    return {OutputFtrType.DETEXT_CLS_PROBABILITIES: tf.nn.softmax(inputs),
            OutputFtrType.DETEXT_CLS_LOGITS: inputs,
            OutputFtrType.DETEXT_CLS_PREDICTED_LABEL: tf.argmax(inputs, axis=-1)}


def _multilabel_classification_output_transform(inputs):
    """
    Transforms the outputs for DeText multi-label classification task.
    :param inputs: Tensor with shape [batch_size, 1, num_classes].
    :return: final output for classification, with shape [batch_size, num_classes]
    """
    # shape: [batch_size, num_classes]
    inputs = tf.squeeze(inputs, axis=-2)
    # Return logits, softmax, and label predictions (as bool tensor) for multi-label classification
    return {OutputFtrType.DETEXT_CLS_PROBABILITIES: tf.nn.sigmoid(inputs),
            OutputFtrType.DETEXT_CLS_LOGITS: inputs,
            # 0.'s Tensor with 1.'s to represent predicted label:
            OutputFtrType.DETEXT_CLS_PREDICTED_LABELS: tf.cast(tf.math.greater(inputs, tf.constant(0.0)), dtype=tf.float32)}


def _binary_classification_output_transform(inputs):
    """
   Transforms the outputs for DeText binary classification task.
   :param inputs: Tensor with shape [batch_size, 1, num_classes].
   :return: final output for classification, with shape [batch_size]
   """
    # shape: [batch_size]
    inputs = tf.squeeze(inputs, axis=[1, 2])
    # Return logits and sigmoid predictions for classification
    return {OutputFtrType.DETEXT_CLS_PROBABILITIES: tf.nn.sigmoid(inputs),
            OutputFtrType.DETEXT_CLS_LOGITS: inputs}


class OutputTransformLayer(tf.keras.layers.Layer):
    """ Output transform layer that prepares the final output based on task types (classification or ranking) """

    def __init__(self, task_type):
        super().__init__()
        self.task_type = task_type

    def call(self, inputs):
        # Handle outputs for different task types
        task_type_to_output_transforms = {
            TaskType.RANKING: _ranking_output_transform,
            TaskType.CLASSIFICATION: _classification_output_transform,
            TaskType.BINARY_CLASSIFICATION: _binary_classification_output_transform,
            TaskType.MULTILABEL_CLASSIFICATION: _multilabel_classification_output_transform
        }
        return task_type_to_output_transforms[self.task_type](inputs)
