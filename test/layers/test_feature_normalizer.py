import tensorflow as tf

from detext.layers import feature_normalizer
from detext.utils.testing.data_setup import DataSetup


class TestFeatureNormalizer(DataSetup, tf.test.TestCase):
    """Unit test for feature_normalizer.py"""

    def testFeatureNormalizer(self):
        batch_size = 10
        list_size = 5
        num_ftrs = 20
        inputs = tf.random.uniform(shape=[batch_size, list_size, num_ftrs])

        ftr_mean = 0.9
        ftr_std = 1.2
        normalizer = feature_normalizer.FeatureNormalizer(ftr_mean, ftr_std)

        self.assertEqual(normalizer.ftr_mean, ftr_mean)
        self.assertEqual(normalizer.ftr_std, ftr_std)
        outputs = normalizer(inputs)
        self.assertAllEqual(outputs, (inputs - ftr_mean) / ftr_std)


if __name__ == '__main__':
    tf.test.main()
