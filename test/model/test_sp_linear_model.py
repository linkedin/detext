import tensorflow as tf

from detext.model import sp_linear_model


class TestSpLinearModel(tf.test.TestCase):
    """Unit test for sp_linear_model."""

    def testSparseTensorMerge(self):
        """Unit test for `sparse_tensor_merge`"""
        wide_ftr_num = 4
        ftr_indices = tf.constant(
            [[1, 0, 0],  # Repeated indices due to padding
             [2, 3, 0]],
            dtype=tf.int32
        )
        zero_num = tf.reduce_sum(tf.cast(tf.equal(ftr_indices, 0), tf.float32), axis=-1) - 1

        ftr_indices = tf.expand_dims(ftr_indices, -1)
        values = tf.ones(shape=tf.shape(ftr_indices))
        shape = tf.ones(shape=[2, 1], dtype=tf.int32) * wide_ftr_num

        wide_ftrs = sp_linear_model.sparse_tensor_merge(ftr_indices, values, shape)

        ftrs_weight = tf.constant([
            [1000],
            [1],
            [2],
            [3]
        ], dtype=tf.float32)
        bias = ftrs_weight[0]

        score = tf.squeeze(tf.sparse.matmul(wide_ftrs, ftrs_weight))
        true_score = score - bias * zero_num
        with self.test_session():
            self.assertAllEqual(score.eval(), [2001., 1005.])
            self.assertAllEqual(zero_num.eval(), [1., 0.])
            self.assertAllEqual(true_score.eval(), [1001., 1005.])

    def testSpLinearModel(self):
        """Unit test for `SparseLinearModel`"""
        # When wide_ftrs_sp_val is None (default to 1)
        wide_ftrs_sp_idx = tf.constant([
            [
                [1, 4, 3, 0],  # Repeated index 0 due to padding. Bias term. Should be added once and only once
                [2, 5, 0, 0],
                [2, 3, 9, 2]
            ],
            [
                [1, 4, 3, 0],
                [2, 5, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        wide_ftrs_sp_val = None
        wide_ftrs_sp_max = 100
        model = sp_linear_model.SparseLinearModel(
            num_wide_sp=wide_ftrs_sp_max,
            wide_ftrs_sp_idx=wide_ftrs_sp_idx,
            wide_ftrs_sp_val=wide_ftrs_sp_val,
            initializer=tf.constant_initializer(1))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(model.score.eval(), [
                [4, 3, 5],
                [4, 3, 1]
            ])

        # When values are all 1, results should be the same
        wide_ftrs_sp_val = tf.ones(tf.shape(wide_ftrs_sp_idx))
        model = sp_linear_model.SparseLinearModel(
            num_wide_sp=wide_ftrs_sp_max,
            wide_ftrs_sp_idx=wide_ftrs_sp_idx,
            wide_ftrs_sp_val=wide_ftrs_sp_val,
            initializer=tf.constant_initializer(1))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(model.score.eval(), [
                [4, 3, 5],
                [4, 3, 1]
            ])

        # When values are floats, results should be the same
        wide_ftrs_sp_val = tf.constant([
            [
                [1, 4, 3, 1],  # Repeated index "1" due to padding. Bias term. Should be added once and only once
                [2, 5, 1, 1],
                [2, 3, 9, 2]
            ],
            [
                [1, 4, 3, 1],
                [2, 5, 1, 1],
                [1, 1, 1, 2]
            ]
        ], dtype=tf.float32)
        model = sp_linear_model.SparseLinearModel(
            num_wide_sp=wide_ftrs_sp_max,
            wide_ftrs_sp_idx=wide_ftrs_sp_idx,
            wide_ftrs_sp_val=wide_ftrs_sp_val,
            initializer=tf.constant_initializer(1))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(model.score.eval(), [
                [9, 8, 17],
                [9, 8, 1]
            ])
