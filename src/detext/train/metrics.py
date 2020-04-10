import tensorflow as tf
import tensorflow_ranking as tfr


def compute_ndcg_tfr(scores, labels, features, topk):
    """Computes NDCG using tf ranking"""
    metric_fn = tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.NDCG, topn=topk)
    return metric_fn(labels, scores, features)


def compute_mrr_tfr(scores, labels, features):
    """Computes MRR using tf ranking

    NOTE: There is a bug in tf-rank that's causing topk to have no effect, so this function is computing MRR for
        whole list WHATEVER topk is.
    """
    metric_fn = tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.MRR)
    return metric_fn(labels, scores, features)


def compute_precision_tfr(scores, labels, features, topk):
    """Computes precision using tf ranking"""
    metric_fn = tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.PRECISION, topn=topk)
    return metric_fn(labels, scores, features)


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
    dcg = actual_score * actual_mask / tf.log(tf.cast(tf.range(2, topk + 2), dtype=tf.float32)) * tf.log(2.0)
    dcg = tf.reduce_sum(dcg, axis=-1)
    return dcg


def compute_mrr(scores, labels, group_size, topk):
    """
    Computes MRR score

    :param scores: Tensor Predicted scores. Shape=[batch_size, max_group_size]
    :param labels: Tensor Labels. Shape=[batch_size, max_group_size]
    :param group_size: Tensor Number of documents in each sample. Shape=[batch_size]
    :param topk: int Number of samples to select in MRR computation
    :return: Metric of average MRR
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

    _, cols = tf.nn.top_k(scores2, k=topk)
    actual_index = get_index(cols)
    actual_score = tf.gather_nd(labels, actual_index)
    # for one query, it is possible that topk is larger than group_size.  The actual_mask makes sure that the padded
    # document get a score of 0 in actual_score.
    actual_mask = tf.gather_nd(mask, actual_index)
    rr = actual_score * actual_mask / tf.cast(tf.range(1, topk + 1), dtype=tf.float32)
    rr = tf.reduce_max(rr, axis=-1)
    return tf.metrics.mean(rr)


def compute_ndcg(scores, labels, group_size, topk):
    """
    Computes ndcg score. Caution: it's the traditional formula instead of the one with numerator as power(2, relevance)

    :param scores: Tensor Predicted scores. Shape=[batch_size, max_group_size]
    :param labels: Tensor Labels. Shape=[batch_size, max_group_size]
    :param group_size: Tensor Number of documents in each sample. Shape=[batch_size]
    :param topk: int Number of samples to select in ndcg computation
    :return: Metric of average ndcg
    """
    idcg = compute_dcg(labels, labels, group_size, topk=topk)
    dcg = compute_dcg(scores, labels, group_size, topk=topk)

    ndcg = tf.div_no_nan(dcg, idcg)
    return tf.metrics.mean(ndcg)


def compute_preat1(scores, labels):
    """
    Compute precision@1, i.e., whether the top1 result has the highest label.
    It should be only applied to one-hot labels.
    :param scores: Tensor Predicted scores. Shape=[batch_size, max_group_size]
    :param labels: Tensor Labels. Shape=[batch_size, max_group_size]
    :return: Metric of average precision@1
    """
    # prediction
    predict = tf.argmax(scores, axis=-1, output_type=tf.int32)
    pred_onehot = tf.one_hot(predict, tf.shape(scores)[-1])
    # gold standard
    max_value = tf.tile(tf.reduce_max(labels, axis=-1, keepdims=True), [1, tf.shape(labels)[-1]])
    is_max_value = tf.cast(tf.equal(labels, max_value), dtype=tf.float32)
    # compute preat1
    preat1 = tf.reduce_sum(pred_onehot * is_max_value, axis=-1)
    return tf.metrics.mean(preat1)


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


def compute_auc(scores, labels):
    """Computes AUC score given scores and labels.

    :param scores: Tensor Predicted scores. Shape=[batch_size, max_group_size].
        max_group_size should be 1
    :param labels: Tensor Labels. Shape=[batch_size, max_group_size]
        max_group_size should be 1
    """
    labels = tf.reshape(labels, shape=[tf.shape(labels)[0]])
    scores = tf.reshape(scores, shape=[tf.shape(scores)[0]])
    prob = tf.sigmoid(scores)
    return tf.metrics.auc(labels, prob, num_thresholds=2000)


def compute_confusion_matrix(scores, labels, num_classes):
    """
    Compute the confusion matrix for multi-class classifications.
    :param scores: Tensor Predicted scores. Shape=[batch_size, num_classes]
    :param labels: Tensor Labels. Shape=[batch_size, max_group_size]
        max_group_size should be 1
    :param num_classes: Number of classes for multi-class classification.
    :return: THe confusion matrix metric in the format of (metric_value, update_op) tuple.
    """
    labels = tf.squeeze(labels, -1)
    probabilities = tf.nn.softmax(scores)
    predicted_indices = tf.argmax(probabilities, 1)

    con_matrix = tf.confusion_matrix(labels=labels, predictions=predicted_indices,
                                     num_classes=num_classes)

    con_matrix_sum = tf.Variable(tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                                 trainable=False,
                                 name="confusion_matrix_result",
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES])
    update_op = tf.assign_add(con_matrix_sum, con_matrix)
    return tf.convert_to_tensor(con_matrix_sum), update_op


def compute_accuracy(scores, labels):
    """
    Compute the accuracy given the scores and labels.
    :param scores: Tensor Predicted scores. Shape=[batch_size, num_classes]
    :param labels: Tensor Labels. Shape=[batch_size, max_group_size]
        max_group_size should be 1
    :return: The accuracy (% of predicted_indices matches labels)
    """
    labels = tf.squeeze(labels, -1)
    probabilities = tf.nn.softmax(scores)
    predicted_indices = tf.argmax(probabilities, 1)
    return tf.metrics.accuracy(labels, predicted_indices)