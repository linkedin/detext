import tensorflow as tf

from detext.layers import shallow_tower_layer
from detext.utils.parsing_utils import InputFtrType
from detext.utils.testing.data_setup import DataSetup


class TestShallowTowerLayer(tf.test.TestCase, DataSetup):
    """Unit test for shallow_tower_layer.py"""
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

    def test(self):
        inputs = {
            InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES: self.sparse_tensor
        }
        first_embedding = tf.range(10, dtype=tf.float32)
        second_embedding = tf.range(10, 20, dtype=tf.float32)

        expected_result = tf.expand_dims(first_embedding * (self.dense_tensor[0][0][0] + self.dense_tensor[1][0][0]) +
                                         second_embedding * (self.dense_tensor[0][0][1] + self.dense_tensor[1][0][1]), axis=0)

        layer = shallow_tower_layer.ShallowTowerLayer(nums_shallow_tower_sparse_ftrs=self.nums_sparse_ftrs,
                                                      num_classes=self.sparse_embedding_size,
                                                      initializer=self.RangeInitializer())
        outputs = layer(inputs)
        embedding_size = self.sparse_embedding_size
        self.assertAllEqual(tf.shape(outputs), [self.batch_size, embedding_size])
        self.assertAllEqual(outputs, expected_result)


if __name__ == '__main__':
    tf.test.main()
