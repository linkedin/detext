import tensorflow as tf

from detext.layers import feature_rescaler
from detext.utils.testing.data_setup import DataSetup


class TestFeatureRescaler(DataSetup, tf.test.TestCase):
    """Unit test for feature_rescaler.py"""

    def testFeatureRescaler(self):
        batch_size = 10
        list_size = 5
        num_ftrs = 20
        inputs = tf.random.uniform(shape=[batch_size, list_size, num_ftrs])

        rescaler = feature_rescaler.FeatureRescaler(num_ftrs)

        self.assertEqual(rescaler._initial_w, 1.0)
        self.assertEqual(rescaler._initial_b, 0.0)
        outputs = rescaler(inputs)
        self.assertAllEqual(outputs, tf.tanh(inputs * rescaler._initial_w + rescaler._initial_b))


if __name__ == '__main__':
    tf.test.main()
