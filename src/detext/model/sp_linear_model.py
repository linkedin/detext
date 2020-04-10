import tensorflow as tf


class SparseLinearModel:
    """Linear model that performs computation on sparse features"""

    def __init__(self,
                 num_wide_sp: int,
                 wide_ftrs_sp_idx: tf.Tensor,
                 wide_ftrs_sp_val: tf.Tensor = None,
                 initializer=tf.contrib.layers.xavier_initializer()):
        """ Computes a linear score given wide feature indices and values
        If wide_ftrs_sp_val is specified, users should keep consistency between wide_ftrs_sp_idx and wide_ftrs_sp_val
            -- the value of wide_ftrs_sp_idx[i] should be wide_ftrs_sp_val[i].
        CAVEAT: it is required that padding value = 0 for wide_ftrs_sp_idx in data_fn and padding value = 1 for
            wide_ftrs_sp_val since feature index 0 is used as the bias term.

        :param wide_ftrs_sp_idx: Feature indices. Shape=[batch_size, max_group_size, max_wide_ftrs_size]
        :param wide_ftrs_sp_val: Feature values. Shape=[batch_size, max_group_size, max_wide_ftrs_size]
        :param num_wide_sp: Number of unique sparse wide features. This should be LARGER than the maximum of
            features indices. Remember to add at least 1 to max(indices) avoid overflow due to padding
        :param initializer: Weight initializer
        """
        wide_ftrs_sp_idx = tf.cast(wide_ftrs_sp_idx, dtype=tf.float32)
        if wide_ftrs_sp_val is None:  # Default to 1 if values unspecified
            wide_ftrs_sp_val = tf.ones(tf.shape(wide_ftrs_sp_idx), dtype=tf.float32)

        self._num_wide_sp = num_wide_sp

        with tf.variable_scope('wide', reuse=tf.AUTO_REUSE):
            # Feature weights
            self.ftrs_weight = tf.get_variable('wide_ftrs_sp_weight',
                                               shape=[num_wide_sp, 1],
                                               initializer=initializer,
                                               trainable=True)

            # A hack to combine idx and val so that we can process them together in `tf.map_fn` later
            wide_ftrs_sp_idx_with_value = tf.concat([wide_ftrs_sp_idx, wide_ftrs_sp_val], axis=-1)

            # Compute linear score sample-wise
            self.score = tf.map_fn(self._compute_linear_score_per_record, wide_ftrs_sp_idx_with_value,
                                   dtype=tf.float32)

    def _compute_linear_score_per_record(self, wide_ftrs_idx_with_value_per_record):
        """Computes the score for given wide_ftrs_sp_idx and wide_ftrs_sp_val.
        A SparseTensor is created using wide_ftrs_sp_idx and wide_ftrs_sp_val and then multiplied with weights to
            obtain a linear score for each document
        """

        # Split idx and val back
        wide_ftrs_sp_idx, wide_ftrs_sp_val = tf.split(wide_ftrs_idx_with_value_per_record, 2, axis=-1)
        wide_ftrs_sp_idx = tf.cast(wide_ftrs_sp_idx, dtype=tf.int64)

        # Transformation
        shape = tf.ones(shape=[tf.shape(wide_ftrs_sp_idx)[0], 1], dtype=tf.int64) * self._num_wide_sp
        valid_wide_ftrs_idx_mask = tf.cast(tf.not_equal(wide_ftrs_sp_idx, 0), tf.float32)
        wide_ftrs_sp_idx = tf.expand_dims(wide_ftrs_sp_idx, -1)

        # Get sparse feature vector v where v[ftr_idx_i] = ftr_val_i and v[other] = 0
        wide_ftrs_sp_idx = sparse_tensor_merge(wide_ftrs_sp_idx,
                                               wide_ftrs_sp_val * valid_wide_ftrs_idx_mask,
                                               shape)

        # Feature weights
        bias = self.ftrs_weight[0]

        # Compute linear score
        score = tf.squeeze(tf.sparse.matmul(wide_ftrs_sp_idx, self.ftrs_weight), axis=-1) + bias
        return score


def sparse_tensor_merge(indices: tf.Tensor, values: tf.Tensor, shape: tf.Tensor) -> tf.SparseTensor:
    """Creates a SparseTensor from grouped indices, values, and shapes
    Copied from https://stackoverflow.com/questions/42147362/create-sparsetensor-from-dequeued-batch-of-values-indices-shapes

    Args:
      indices: A [batch_size, N, D] integer Tensor
      values: A [batch_size, N] Tensor of any dtype
      shape: A [batch_size, D] Integer Tensor
    Returns:
      A SparseTensor of dimension D + 1 with batch_size as its first dimension.
    """
    indices = tf.cast(indices, dtype=tf.int64)
    shape = tf.cast(shape, dtype=tf.int64)
    merged_shape = tf.reduce_max(shape, axis=0)
    batch_size, elements, shape_dim = tf.unstack(tf.shape(indices))
    index_range_tiled = tf.tile(tf.range(batch_size)[..., None],
                                tf.stack([1, elements]))[..., None]
    merged_indices = tf.reshape(
        tf.concat([tf.cast(index_range_tiled, tf.int64), indices], axis=2),
        [-1, 1 + tf.size(merged_shape)])
    merged_values = tf.reshape(values, [-1])
    concat_shape = tf.concat([[tf.cast(batch_size, tf.int64)], merged_shape], axis=0)

    return tf.SparseTensor(merged_indices, merged_values, concat_shape)
