import tensorflow as tf


class FeatureRescaler(tf.keras.layers.Layer):
    """Feature rescaler to rescale dense features

    This layer improves numeric stability and is useful for network convergence
    """

    def __init__(self, num_ftrs, prefix=''):
        super(FeatureRescaler, self).__init__()
        self._num_ftrs = num_ftrs
        self._initial_w = 1.0
        self._initial_b = 0.0

        self.norm_w = tf.compat.v1.get_variable(f"{prefix}norm_w", [num_ftrs], dtype=tf.float32,
                                                initializer=tf.compat.v1.constant_initializer(self._initial_w))
        self.norm_b = tf.compat.v1.get_variable(f"{prefix}norm_b", [num_ftrs], dtype=tf.float32,
                                                initializer=tf.compat.v1.constant_initializer(self._initial_b))

    def call(self, inputs, **kwargs):
        """ Rescales inputs to tf.tanh(inputs * self.norm_w + self.norm_b)

        :param inputs: Tensor(tf.float32). Shape=[..., num_ftrs]
        :param kwargs: Dummy args for suppress warning for method overriding
        :return: Rescaled input
        """
        return tf.tanh(inputs * self.norm_w + self.norm_b)
