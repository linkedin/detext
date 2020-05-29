import numpy as np
import tensorflow as tf
from detext.model import sp_emb_model


class TestSpEmbModel(tf.test.TestCase):
    """Unit test for sp_emb_model"""

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
        wide_ftrs = sp_emb_model.sparse_tensor_merge(ftr_indices, values, shape)
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

    def testSpEmbModel(self):
        test_data = list()
        for sp_emb_size in [1, 100]:
            # When wide_ftrs_sp_val is None (default to 1)
            test_data.append(
                {
                    "wide_ftrs_sp_idx": [
                        [
                            [1, 4, 3, 0],
                            # Repeated index 0 due to padding. Bias term. Should be added once and only once
                            [2, 5, 0, 0],
                            [2, 3, 9, 2]
                        ],
                        [
                            [1, 4, 3, 0],
                            [2, 5, 0, 0],
                            [0, 0, 0, 0]
                        ]
                    ],
                    "wide_ftrs_sp_val": None,
                    "wide_ftrs_sp_max": 100,
                    "sp_emb_size": sp_emb_size,
                    "ground_truth": np.expand_dims([
                        [4, 3, 5],
                        [4, 3, 1]
                    ], axis=-1)
                }
            )
            # When values are all 1, results should be the same
            test_data.append(
                {
                    "wide_ftrs_sp_idx": [
                        [
                            [1, 4, 3, 0],
                            # Repeated index 0 due to padding. Bias term. Should be added once and only once
                            [2, 5, 0, 0],
                            [2, 3, 9, 2]
                        ],
                        [
                            [1, 4, 3, 0],
                            [2, 5, 0, 0],
                            [0, 0, 0, 0]
                        ]
                    ],
                    "wide_ftrs_sp_val": np.ones([2, 3, 4]),
                    "wide_ftrs_sp_max": 100,
                    "sp_emb_size": sp_emb_size,
                    "ground_truth": np.expand_dims([
                        [4, 3, 5],
                        [4, 3, 1]
                    ], axis=-1)
                }
            )
            # When values are random floats
            test_data.append({
                "wide_ftrs_sp_idx":
                    [
                        [
                            [1, 4, 3, 0],
                            # Repeated index 0 due to padding. Bias term. Should be added once and only once
                            [2, 5, 0, 0],
                            [2, 3, 9, 2]
                        ],
                        [
                            [1, 4, 3, 0],
                            [2, 5, 0, 0],
                            [0, 0, 0, 0]
                        ]
                    ],
                "wide_ftrs_sp_val":
                    [
                        [
                            [1, 4, 3, 1],
                            [2, 5, 1, 1],
                            [2, 3, 9, 2]
                        ],
                        [
                            [1, 4, 3, 1],
                            [2, 5, 1, 1],
                            [1, 1, 1, 2]
                        ]
                    ],
                "wide_ftrs_sp_max": 100,
                "sp_emb_size": sp_emb_size,
                "ground_truth": np.expand_dims([
                    [9, 8, 17],
                    [9, 8, 1]
                ], axis=-1)
            })
        for input in test_data:
            self._testSpEmbModel(input)

    def _testSpEmbModel(self, input):
        """Unit test for `SpEmbModel`"""
        graph = tf.Graph()
        with graph.as_default():
            wide_ftrs_sp_idx = tf.constant(input["wide_ftrs_sp_idx"])
            wide_ftrs_sp_val = input["wide_ftrs_sp_val"]
            if wide_ftrs_sp_val is not None:
                wide_ftrs_sp_val = tf.constant(wide_ftrs_sp_val, dtype=tf.float32)
            wide_ftrs_sp_max = input["wide_ftrs_sp_max"]
            ground_truth = input["ground_truth"]
            sp_emb_size = input["sp_emb_size"]
            model = sp_emb_model.SparseEmbModel(
                num_wide_sp=wide_ftrs_sp_max,
                wide_ftrs_sp_idx=wide_ftrs_sp_idx,
                wide_ftrs_sp_val=wide_ftrs_sp_val,
                sp_emb_size=sp_emb_size,
                initializer=tf.constant_initializer(1))
        with self.test_session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(model.embedding.eval(), np.repeat(ground_truth, sp_emb_size, axis=-1))


if __name__ == "__main__":
    tf.test.main()
