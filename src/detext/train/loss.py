import tensorflow as tf
import tensorflow_ranking as tfr

from detext.train.constant import Constant
from detext.utils.parsing_utils import TaskType


class LambdaRank(tf.compat.v1.layers.Layer):
    """
    LambdaRank as the learning-to-rank function.
    For now, the current version does not implement the "lambda" part.
    """

    @staticmethod
    def compute_rank(scores, group_size):
        """
        compute each element's rank
        """
        num_cols_scores = tf.shape(scores)[1]
        # Make sure the padded docs have lowest scores
        mask = tf.sequence_mask(group_size, maxlen=num_cols_scores, dtype=tf.float32)
        scores2 = (scores - tf.reduce_min(scores)) * mask - (1 - mask)
        # construct rank matrix using tf.nn.top_k
        _, col_idx = tf.nn.top_k(scores2, k=num_cols_scores)
        col_idx = tf.reshape(col_idx, shape=[-1])
        row_idx = tf.range(tf.size(scores)) // num_cols_scores
        idx = tf.concat([tf.expand_dims(row_idx, -1), tf.expand_dims(col_idx, -1)], axis=1)
        val = tf.cast(tf.range(tf.size(scores)) % num_cols_scores + 1, dtype=tf.float32)
        rank_mat = tf.scatter_nd(idx, val, shape=tf.shape(scores))
        return rank_mat

    @staticmethod
    def compute_dcg(scores, labels, group_size, topk):
        """
        Compute the dcg score.
        """
        max_group_size = tf.reduce_max(group_size)
        batch_size = tf.shape(scores)[0]
        topk = tf.minimum(max_group_size, topk)
        # The padded docs should have lowest scores
        mask = tf.sequence_mask(group_size, maxlen=tf.shape(scores)[1], dtype=tf.float32)
        scores2 = (scores - tf.reduce_min(scores)) * mask - (1 - mask)

        def get_index(cols):
            rows = tf.reshape(tf.range(batch_size * topk) // topk, [batch_size, topk])
            index = tf.stack([rows, cols], axis=2)
            return index

        # dcg
        _, cols = tf.nn.top_k(scores2, k=topk)
        actual_index = get_index(cols)
        actual_score = tf.gather_nd(labels, actual_index)
        # for one query, it is possible that topk is larger than group_size.  The actual_mask makes sure that the padded
        # document get a score of 0 in actual_score.
        actual_mask = tf.gather_nd(mask, actual_index)
        dcg = actual_score * actual_mask / tf.math.log(tf.cast(tf.range(2, topk + 2), dtype=tf.float32)) * tf.math.log(2.0)
        dcg = tf.reduce_sum(dcg, axis=-1)
        return dcg

    def __init__(self, lambda_metric=None):
        self.lambda_metric = lambda_metric
        super(LambdaRank, self).__init__()

    def build(self, _):
        super(LambdaRank, self).build(_)

    def call(self, scores, labels):
        """ Compute the pairwise loss.

        :param scores: A tensor with shape [batch_size, max_group_size].  For each batch, the first element is the score of
        correct answer.
        :param labels: A matrix with shape [batch_size, max_group_size].  The true scores of each document.
        :return: lambdarank loss and mask. Each with shape [batch_size, max_group_size, max_group_size]
        """

        # for a query, compute the pairwise doc score diff of any two documents.
        pair_score_diff = tf.expand_dims(scores, axis=2) - tf.expand_dims(scores, axis=1)
        # compute the loss
        loss = -1 * tf.math.log_sigmoid(pair_score_diff)
        # now loss is a [batch_size, max_group_size, max_group_size] tensor that contains all pairwise loss.
        # we only need to keep a subset of the pairs.
        # the first mask is from group_size
        group_size_mask = tf.cast(labels != Constant()._LABEL_PADDING, dtype=tf.float32)
        group_size = tf.reduce_sum(group_size_mask, axis=-1)
        group_size_mask = tf.expand_dims(group_size_mask, axis=2) * tf.expand_dims(group_size_mask, axis=1)
        # the second mask is from label; only keep the pairs that 1st label value is larger than 2nd value
        label_mask = tf.expand_dims(labels, axis=2) - tf.expand_dims(labels, axis=1)
        label_mask = tf.cast(tf.greater(label_mask, tf.zeros_like(label_mask)), dtype=tf.float32)
        pairwise_mask = group_size_mask * label_mask
        loss *= pairwise_mask

        if self.lambda_metric:
            # compute each element's rank
            rank_mat = self.compute_rank(scores, group_size)
            if self.lambda_metric['metric'] == 'ndcg':
                # ideal dcg
                idcg = self.compute_dcg(labels, labels, group_size, self.lambda_metric['topk'])
                # delta_score
                delta_score = tf.expand_dims(labels, axis=2) - tf.expand_dims(labels, axis=1)
                # delta rank
                reci_log_rank = tf.math.log(2.0) / tf.math.log(tf.cast(rank_mat, dtype=tf.float32) + 1)
                delta_rank = tf.expand_dims(reci_log_rank, axis=2) - tf.expand_dims(reci_log_rank, axis=1)
                # delta_ndcg = |delta_score * delta_rank| / idcg
                delta_ndcg = tf.abs(delta_score * delta_rank) / tf.expand_dims(tf.expand_dims(idcg, 1), 1)
                # lambda loss
                loss *= delta_ndcg

        return loss, pairwise_mask


def compute_softmax_loss(scores, labels):
    """
    It computes the sum of negative log softmax loss:
    -sum_i lables_i * log(exp(scores_i) / (exp(scores_1) + ... + exp(scores_n)))
    """
    # mask the padded documents
    mask = tf.cast(labels != Constant()._LABEL_PADDING, dtype=tf.float32)
    # softmax loss
    loss = mask * labels * (-scores + tf.expand_dims(compute_logsumexp_mask(scores, mask), axis=1))
    return loss


def compute_sigmoid_cross_entropy_loss(scores, labels):
    """ Compute loss for pointwise ranking

    :param scores: Tensor  Shape=[batch_size, max_group_size]
    :param labels: Tensor  Shape=[batch_size, max_group_size]
    :param group_size: Tensor  Shape=[batch_size]
    :return: Tensor  Shape=[batch_size, max_group_size]
   """
    mask = tf.cast(labels != Constant()._LABEL_PADDING, dtype=tf.float32)
    loss = mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=scores)
    return tf.reduce_sum(input_tensor=loss, axis=-1)


def compute_logsumexp_mask(scores, mask):
    """
    Compute logsumexp with mask for a batch.
    Current tf.lossumexp does not support mask.
    """
    # Change padding scores to be the score of the first element in the score array. I.e., if
    #   original_scores = [[1., -1., PAD_SCORE], [2., -2., PAD_SCORE]], result_scores will be
    #   [[1., -1., 1.], [2., -2., 2]]. This is to make sure the maximum value is chosen from
    #   VALID scores
    scores = tf.compat.v1.where(tf.cast(mask, dtype=tf.bool), scores,
                                tf.ones_like(scores) * tf.expand_dims(scores[:, 0], axis=-1))
    max_scores = tf.reduce_max(input_tensor=scores, axis=-1, keepdims=True)
    scores = scores - max_scores
    exp_score_withmask = tf.exp(scores) * tf.cast(mask, dtype=tf.float32)
    logsumexp = tf.math.log(tf.reduce_sum(input_tensor=exp_score_withmask, axis=-1)) + tf.squeeze(max_scores, -1)
    return logsumexp


def compute_regularization_penalty(l1, l2, trainable_vars):
    """ Returns the regularization penalty specified in hparams """
    l1 = l1 if l1 is not None else 0
    l2 = l2 if l2 is not None else 0
    regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    penalty = 0.0
    for weight in trainable_vars:
        penalty += regularizer(weight)
    return penalty


def compute_loss(task_type, ltr_loss_fn, tfr_loss_fn, tfr_lambda_weights, use_tfr_loss, l1, l2, scores, labels, weight, trainable_vars):
    """ Computes loss with regularization based on task_type for selected task"""
    if weight is None:
        raise ValueError("weight should not be None")

    task_type_to_loss_fn = {
        TaskType.RANKING:
            lambda scores, labels, weight, ltr_loss_fn, tfr_loss_fn, tfr_lambda_weights, use_tfr_loss:
            compute_rank_loss(scores, labels, weight, ltr_loss_fn, tfr_loss_fn, tfr_lambda_weights, use_tfr_loss),
        TaskType.CLASSIFICATION:
            lambda scores, labels, weight, *args:
            compute_classification_loss(scores, labels, weight),
        TaskType.BINARY_CLASSIFICATION:
            lambda scores, labels, weight, *args:
            compute_binary_classification_loss(scores, labels, weight),
    }

    task_loss = task_type_to_loss_fn[task_type](scores, labels, weight, ltr_loss_fn, tfr_loss_fn, tfr_lambda_weights, use_tfr_loss)
    return task_loss + tf.reduce_mean(input_tensor=weight) * compute_regularization_penalty(l1, l2, trainable_vars)


def compute_rank_loss(scores, labels, weight, ltr_loss_fn, tfr_loss_fn, tfr_lambda_weights, use_tfr_loss):
    """
    Compute ranking loss
    Note that the tfr loss is slightly different than our implementation: the tfr loss is sum over all loss and
    devided by number of queries; our implementation is sum over all loss and devided by the number of larger than
    0 labels.
    """
    # Expand weight to [batch size, 1] so that in inhouse ranking loss it can be multiplied with loss which is
    #   [batch_size, max_group_size]
    expanded_weight = tf.expand_dims(weight, axis=-1)

    # tf-ranking loss
    if use_tfr_loss:
        weight_name = "weight"
        loss_fn = tfr.losses.make_loss_fn(tfr_loss_fn, lambda_weight=tfr_lambda_weights,
                                          weights_feature_name=weight_name)
        loss = loss_fn(labels, scores, {weight_name: expanded_weight})
        return loss

    # our own implementation
    if ltr_loss_fn == 'pairwise':
        lambdarank = LambdaRank()
        pairwise_loss, pairwise_mask = lambdarank(scores, labels)
        loss = tf.reduce_sum(
            input_tensor=tf.reduce_sum(input_tensor=pairwise_loss, axis=[1, 2]) * expanded_weight) / tf.reduce_sum(
            input_tensor=pairwise_mask)
    elif ltr_loss_fn == 'softmax':
        loss = compute_softmax_loss(scores, labels) * expanded_weight
        is_positive_label = tf.cast(tf.greater(labels, 0), dtype=tf.float32)
        loss = tf.math.divide_no_nan(tf.reduce_sum(input_tensor=loss), tf.reduce_sum(input_tensor=is_positive_label))
    elif ltr_loss_fn == 'pointwise':
        loss = compute_sigmoid_cross_entropy_loss(scores, labels) * expanded_weight
        loss = tf.reduce_mean(input_tensor=loss)
    else:
        raise ValueError('Currently only support pointwise/pairwise/softmax/softmax_cls.')
    return loss


def compute_classification_loss(scores, labels, weight):
    """ Compute classification loss

    :param scores Shape=[batch_size, num_classes]
    :param labels Shape=[batch_size]
    :param weight Shape=[batch_size]
    """
    # Classification loss
    labels = tf.cast(labels, tf.int32)
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=scores, labels=labels, weights=weight)


def compute_binary_classification_loss(scores, labels, weight):
    """ Compute classification loss

    :param scores Shape=[batch_size]
    :param labels Shape=[batch_size]
    :param weight Shape=[batch_size]
    """
    # Binary classification loss
    return tf.compat.v1.losses.sigmoid_cross_entropy(logits=tf.expand_dims(scores, axis=1),
                                                     multi_class_labels=tf.expand_dims(labels, axis=1),
                                                     weights=tf.expand_dims(weight, axis=1))
