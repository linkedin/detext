from typing import List

import tensorflow as tf


class MultiLayerPerceptron(tf.keras.layers.Layer):
    """ A multi layer perceptron """

    def __init__(self, num_hidden: List[int], activations: List, prefix: str = ''):
        """ Initializes the layer

        :param num_hidden: list of hidden layer sizes
        :param activations: list of activations for dense layer
        :param prefix: prefix of hidden layer name
        """
        super(MultiLayerPerceptron, self).__init__()
        assert len(num_hidden) == len(activations), "num hidden and activations must contain the same number of elements"

        self.mlp = []
        for i, (hidden_size, activation) in enumerate(zip(num_hidden, activations)):
            if hidden_size == 0:
                continue
            layer = tf.keras.layers.Dense(units=hidden_size, use_bias=True, activation=activation,
                                          name=f'{prefix}hidden_projection_{str(i)}')
            self.mlp.append(layer)

    def call(self, inputs, **kwargs):
        """ Applies multi-layer perceptron on given inputs

        :return output Shape=inputs.shape[:-1] + [num_hidden[-1]]
        """
        x = inputs
        for layer in self.mlp:
            x = layer(x)
        return x
