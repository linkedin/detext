from typing import List

import tensorflow as tf

from .losses import compute_text_generation_loss


class NegativePerplexity(tf.keras.metrics.Metric):
    """ Metric for computing perplexity"""

    def __init__(self, **kwargs):
        super(NegativePerplexity, self).__init__(**kwargs)
        self.cross_entropy_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.cross_entropy_loss_val = self.add_weight(name='total_perplexity', initializer='zeros')
        self.num_samples = self.add_weight(name='sample_count', initializer='zeros')

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def result(self):
        return -tf.math.exp(self.cross_entropy_loss_val)

    def update_state(self, labels, logits, lengths, sample_weight=None):
        """ Accumulates metric statistics

        :param logits: Tensor Predicted scores. Shape=[batch_size, max_sentence_length, vocab_size]
        :param labels: Tensor Labels. Shape=[batch_size, max_sentence_length]
        :param sample_weight: Sample weight. Check the inherited class method for more detail
        """
        loss = compute_text_generation_loss(logits=logits, labels=labels, lengths=lengths)
        batch_size = tf.cast(tf.shape(labels)[0], dtype=tf.dtypes.float32)

        self.cross_entropy_loss_val.assign(
            self.cross_entropy_loss_val * (self.num_samples / (self.num_samples + batch_size)) +
            loss * batch_size / (self.num_samples + batch_size))
        self.num_samples.assign_add(batch_size)


def get_metric_fn(metric_name):
    """ Returns the corresponding metric_fn according to metric name"""
    metrics = {'perplexity': lambda: NegativePerplexity(name=metric_name)}

    # Metric not found in ranking metric. Switch to classification metric matching
    for clf_metric_name, metric_fn in metrics.items():
        if metric_name == clf_metric_name:
            return metric_fn

    raise ValueError(f'Unsupported metric name: {metric_name}')


def get_metric_fn_lst(all_metrics: List[str]):
    """ Returns a list of metric_fn from the given metrics

    :param all_metrics A list of metrics supported by Smart Compose
    """
    metric_fn_lst = []

    for metric_name in all_metrics:
        metric_fn_lst.append(get_metric_fn(metric_name))

    return metric_fn_lst
