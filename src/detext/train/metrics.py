from typing import List

import tensorflow as tf
import tensorflow_ranking as tfr

from detext.utils.parsing_utils import TaskType

_TOPK_DELIMITER = '@'
_DEFAULT_TOPK = 10
_DEFAULT_AUC_NUM_THRESHOLD = 2000


class BinaryAccuracyMetric(tf.keras.metrics.BinaryAccuracy):
    """ Metric for computing BinaryAccuracy given DeText format input """

    def update_state(self, labels, scores, sample_weight=None):
        """ Accumulates metric statistics

        :param scores: Tensor Predicted scores. Shape=[batch_size]
        :param labels: Tensor Labels. Shape=[batch_size]
        :param sample_weight: Sample weight. Check the inherited class method for more detail
        """
        scores = tf.nn.sigmoid(scores)
        return super().update_state(labels, scores, sample_weight)


class AccuracyMetric(tf.keras.metrics.SparseCategoricalAccuracy):
    """ Metric for computing Accuracy given DeText format input """

    def update_state(self, labels, scores, sample_weight=None):
        """ Accumulates metric statistics

        :param scores: Tensor Predicted scores. Shape=[batch_size, num_classes]
        :param labels: Tensor Labels. Shape=[batch_size]
        :param sample_weight: Sample weight. Check the inherited class method for more detail
        """
        labels = tf.expand_dims(labels, axis=1)
        return super(AccuracyMetric, self).update_state(labels, scores, sample_weight)


class AUCMetric(tf.keras.metrics.AUC):
    """ Metric for computing AUC given DeText format input """

    def update_state(self, labels, scores, sample_weight=None):
        """ Accumulates metric statistics

        :param scores: Tensor Predicted scores. Shape=[batch_size, max_group_size(1)] for ranking. For classification, shape=[batch_size]
        :param labels: Tensor Labels. Shape=[batch_size, max_group_size(1)] for ranking. For binary classification, shape=[batch_size]
        :param sample_weight: Sample weight. Check the inherited class method for more detail
        """
        labels = tf.reshape(labels, shape=[tf.shape(input=labels)[0]])
        scores = tf.reshape(scores, shape=[tf.shape(input=scores)[0]])
        prob = tf.sigmoid(scores)
        return super(AUCMetric, self).update_state(labels, prob, sample_weight)


class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    Metric for computing confusion matrix
    Credit to https://towardsdatascience.com/custom-metrics-in-keras-and-how-simple-they-are-to-use-in-tensorflow2-2-6d079c2ca279
    """

    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrixMetric, self).__init__(**kwargs)  # handles base args (e.g., dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer='zeros')

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, labels, scores, sample_weight=None):
        """ Accumulates metric statistics

        :param scores: Tensor Predicted scores. Shape=[batch_size, num_classes]
        :param labels: Tensor Labels. Shape=[batch_size, max_group_size]
            max_group_size should be 1
        :param sample_weight: Sample weight. Check the inherited class method for more detail
        """
        scores = tf.argmax(scores, 1)
        cm = tf.math.confusion_matrix(labels, scores, dtype=tf.float32, num_classes=self.num_classes)
        self.total_cm.assign_add(cm)
        return self.total_cm

    def result(self):
        return self.total_cm


def get_metric_fn(metric_name, task_type, num_classes):
    """ Returns the corresponding metric_fn according to metric name"""

    topk = int(metric_name.split(_TOPK_DELIMITER)[1]) if _TOPK_DELIMITER in metric_name else _DEFAULT_TOPK

    ranking_metrics = {'ndcg': lambda: tfr.keras.metrics.NDCGMetric(topn=topk, name=metric_name),
                       'mrr': lambda: tfr.keras.metrics.MRRMetric(topn=topk, name=metric_name),
                       'precision': lambda: tfr.keras.metrics.PrecisionMetric(topn=topk, name=metric_name),
                       'auc': lambda: AUCMetric(num_thresholds=_DEFAULT_AUC_NUM_THRESHOLD, name=metric_name)}

    classification_metrics = {
        'accuracy': lambda: AccuracyMetric(name=metric_name) if task_type == TaskType.CLASSIFICATION else BinaryAccuracyMetric(name=metric_name),
        'confusion_matrix': lambda: ConfusionMatrixMetric(num_classes=num_classes, name=metric_name),
        'auc': lambda: AUCMetric(num_thresholds=_DEFAULT_AUC_NUM_THRESHOLD, name=metric_name)
    }

    # Add ranking metric function
    if task_type == TaskType.RANKING:
        for prefix, metric_fn in ranking_metrics.items():
            if metric_name.startswith(prefix):
                if prefix == 'auc':
                    assert metric_name == 'auc', 'AUC metric requires exact match'
                    return metric_fn

                return metric_fn
        raise ValueError(f'Unsupported metric name: {metric_name}')

    # Metric not found in ranking metric. Switch to classification metric matching
    if task_type in [TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION]:
        assert num_classes is not None and num_classes > 0, f'num_classes has to be positive integer. Current num_classes: {num_classes}'
        for clf_metric_name, metric_fn in classification_metrics.items():
            if metric_name == clf_metric_name:
                return metric_fn
        raise ValueError(f'Unsupported metric name: {metric_name}')

    raise ValueError(f'Unsupported task type: {task_type}')


def get_metric_fn_lst(all_metrics: List[str], task_type, num_classes: int):
    """ Returns a list of metric_fn from the given metrics

    :param all_metrics A list of metrics supported by DeText
    :param task_type Type of task. E.g., ranking/classification/binary_classification
    :param num_classes Number of classes in the classification task
    """
    metric_fn_lst = []

    for metric_name in all_metrics:
        metric_fn_lst.append(get_metric_fn(metric_name, task_type, num_classes))

    return metric_fn_lst
