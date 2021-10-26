import tensorflow as tf

from detext.layers import sparse_embedding_layer
from detext.utils.parsing_utils import InputFtrType
from detext.utils.testing.data_setup import DataSetup


class TestSparseEmbeddingLayer(tf.test.TestCase, DataSetup):
    """Unit test for sparse_embedding_layer.py"""
    dense_tensor = [tf.constant([[0, 3]], dtype=tf.dtypes.float32),
                    tf.constant([[4, 20]], dtype=tf.dtypes.float32)]
    sparse_tensor = [tf.sparse.from_dense(t) for t in dense_tensor]

    sparse_embedding_size = 10
    nums_sparse_ftrs = [2, 2]
    batch_size = tf.shape(dense_tensor[0])[0]

    class RangeInitializer(tf.keras.initializers.Initializer):
        def __init__(self):
            super().__init__()

        def __call__(self, shape, dtype=None, **kwargs):
            return tf.reshape(tf.range(0, tf.reduce_prod(shape), dtype=dtype), shape=shape)

    def testConcat(self):
        self._testConcatRangeInitializer()
        self._testConcatOnesInitializer()

    def _testConcatOnesInitializer(self):
        embedding = tf.ones([1, self.sparse_embedding_size], dtype=tf.dtypes.float32)
        expected_result_list = [tf.concat([embedding * tf.reduce_sum(self.dense_tensor[0][0]),
                                           embedding * tf.reduce_sum(self.dense_tensor[1][0])], axis=-1),
                                tf.concat([embedding, embedding], axis=-1)]

        combiner_list = ['sum', 'mean']
        for combiner, expected_result in zip(combiner_list, expected_result_list):
            self._testInput(combiner, 'concat', expected_result, 'ones')

    def _testConcatRangeInitializer(self):
        first_embedding = tf.range(10, dtype=tf.float32)
        second_embedding = tf.range(10, 20, dtype=tf.float32)

        expected_result = tf.expand_dims(
            tf.concat([first_embedding * self.dense_tensor[0][0][0], first_embedding * self.dense_tensor[1][0][0]], axis=-1) +
            tf.concat([second_embedding * self.dense_tensor[0][0][1], second_embedding * self.dense_tensor[1][0][1]], axis=-1),
            axis=0)
        self._testInput('sum', 'concat', expected_result, self.RangeInitializer())

    def testSum(self):
        self._testSumRangeInitializer()
        self._testSumOnesInitializer()

    def _testSumRangeInitializer(self):
        first_embedding = tf.range(10, dtype=tf.float32)
        second_embedding = tf.range(10, 20, dtype=tf.float32)

        expected_result = tf.expand_dims(first_embedding * (self.dense_tensor[0][0][0] + self.dense_tensor[1][0][0]) +
                                         second_embedding * (self.dense_tensor[0][0][1] + self.dense_tensor[1][0][1]), axis=0)
        self._testInput('sum', 'sum', expected_result, self.RangeInitializer())

    def _testSumOnesInitializer(self):
        expected_result_list = [tf.ones([1, self.sparse_embedding_size], dtype=tf.dtypes.float32) * tf.reduce_sum(self.dense_tensor),
                                tf.ones([1, self.sparse_embedding_size], dtype=tf.dtypes.float32) * len(self.dense_tensor)]

        combiner_list = ['sum', 'mean']
        for combiner, expected_result in zip(combiner_list, expected_result_list):
            self._testInput(combiner, 'sum', expected_result, 'ones')

    def _testInput(self, sparse_embedding_same_ftr_combiner, sparse_embedding_cross_ftr_combiner, expected_result, initializer):
        inputs = {
            InputFtrType.SPARSE_FTRS_COLUMN_NAMES: self.sparse_tensor
        }
        layer = sparse_embedding_layer.SparseEmbeddingLayer(sparse_embedding_size=self.sparse_embedding_size,
                                                            nums_sparse_ftrs=self.nums_sparse_ftrs,
                                                            initializer=initializer, sparse_embedding_cross_ftr_combiner=sparse_embedding_cross_ftr_combiner,
                                                            sparse_embedding_same_ftr_combiner=sparse_embedding_same_ftr_combiner)
        outputs = layer(inputs)
        embedding_size = {'sum': self.sparse_embedding_size, 'concat': self.sparse_embedding_size * len(self.nums_sparse_ftrs)}[
            sparse_embedding_cross_ftr_combiner]
        self.assertAllEqual(tf.shape(outputs), [self.batch_size, embedding_size])
        self.assertAllEqual(outputs, expected_result)


if __name__ == '__main__':
    tf.test.main()
