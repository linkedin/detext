"""
lambda rank implementation.
"""
import tensorflow as tf

from detext.train import metrics


class LambdaRank(tf.layers.Layer):
    """
    LambdaRank as the learning-to-rank function.
    For now, the current version does not implement the "lambda" part.
    """

    def __init__(self, lambda_metric=None):
        self.lambda_metric = lambda_metric
        super(LambdaRank, self).__init__()

    def build(self, _):
        super(LambdaRank, self).build(_)

    def call(self, scores, labels, group_size):
        """ Compute the pairwise loss.

        :param scores: A tensor with shape [batch_size, max_group_size].  For each batch, the first element is the score of
        correct answer.
        :param labels: A matrix with shape [batch_size, max_group_size].  The true scores of each document.
        :param group_size: A vector with shape [batch_size], indicating how many documents each query has.
        :return: lambdarank loss and mask. Each with shape [batch_size, max_group_size, max_group_size]
        """
        # for a query, compute the pairwise doc score diff of any two documents.
        pair_score_diff = tf.expand_dims(scores, axis=2) - tf.expand_dims(scores, axis=1)
        # compute the loss
        loss = -1 * tf.log_sigmoid(pair_score_diff)
        # now loss is a [batch_size, max_group_size, max_group_size] tensor that contains all pairwise loss.
        # we only need to keep a subset of the pairs.
        # the first mask is from group_size
        group_size_mask = tf.sequence_mask(group_size, maxlen=tf.shape(scores)[1], dtype=tf.float32)
        group_size_mask = tf.expand_dims(group_size_mask, axis=2) * tf.expand_dims(group_size_mask, axis=1)
        # the second mask is from label; only keep the pairs that 1st label value is larger than 2nd value
        label_mask = tf.expand_dims(labels, axis=2) - tf.expand_dims(labels, axis=1)
        label_mask = tf.cast(tf.greater(label_mask, tf.zeros_like(label_mask)), dtype=tf.float32)
        pairwise_mask = group_size_mask * label_mask
        loss *= pairwise_mask

        if self.lambda_metric:
            # compute each element's rank
            rank_mat = metrics.compute_rank(scores, group_size)
            if self.lambda_metric['metric'] == 'ndcg':
                # ideal dcg
                idcg = metrics.compute_dcg(labels, labels, group_size, self.lambda_metric['topk'])
                # delta_score
                delta_score = tf.expand_dims(labels, axis=2) - tf.expand_dims(labels, axis=1)
                # delta rank
                reci_log_rank = tf.log(2.0) / tf.log(tf.cast(rank_mat, dtype=tf.float32) + 1)
                delta_rank = tf.expand_dims(reci_log_rank, axis=2) - tf.expand_dims(reci_log_rank, axis=1)
                # delta_ndcg = |delta_score * delta_rank| / idcg
                delta_ndcg = tf.abs(delta_score * delta_rank) / tf.expand_dims(tf.expand_dims(idcg, 1), 1)
                # lambda loss
                loss *= delta_ndcg

        return loss, pairwise_mask
