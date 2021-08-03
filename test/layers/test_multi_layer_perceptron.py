import tensorflow as tf

from detext.layers import multi_layer_perceptron


class TestMultiLayerPerceptron(tf.test.TestCase):
    """Unit test of multi_layer_perceptron.py"""

    def testMultiLayerPerceptron(self):
        """Tests MultiLayerPerceptron class"""
        num_hidden = [3, 5, 1]
        activations = ['tanh'] * len(num_hidden)
        prefix = ''
        layer = multi_layer_perceptron.MultiLayerPerceptron(num_hidden, activations, prefix)

        input_shape = [2, 3, 4]
        x = tf.random.uniform(input_shape)
        y = layer(x)
        self.assertEqual(y.shape, input_shape[:-1] + num_hidden[-1:])


if __name__ == '__main__':
    tf.test.main()
