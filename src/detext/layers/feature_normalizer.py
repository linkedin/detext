import tensorflow as tf


class FeatureNormalizer(tf.keras.layers.Layer):
    """Feature normalizer to normalize dense features

    This layer improves numeric stability and is useful for network convergence
    """

    def __init__(self, ftr_mean, ftr_std):
        super(FeatureNormalizer, self).__init__()
        self.ftr_mean = tf.constant(ftr_mean, dtype=tf.dtypes.float32)
        self.ftr_std = tf.constant(ftr_std, dtype=tf.dtypes.float32)

    def call(self, inputs, **kwargs):
        """ Normalizes inputs to (inputs - self.ftr_mean) / self.ftr_std

        :param inputs: Tensor(tf.float32). Shape=[..., num_ftrs]
        :param kwargs: Dummy args for suppress warning for method overriding
        :return: Normalized input
        """
        return (inputs - self.ftr_mean) / self.ftr_std
