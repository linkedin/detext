from abc import ABC, abstractmethod

import tensorflow as tf

from smart_compose.utils import layer_utils
from smart_compose.utils.testing.test_case import TestCase


class TestLayerUtils(TestCase):
    """Unit test for layer_utils.py"""

    def testExpandToSameRank(self):
        """Tests expand_to_same_rank() """
        a = tf.ones([1, 2])
        b = tf.ones([1, 2, 3, 4])
        expanded_a = layer_utils.expand_to_same_rank(a, b)
        self.assertAllEqual(tf.shape(expanded_a), [1, 2, 1, 1])

    def testGetLastValidElements(self):
        """Tests get_last_valid_elements() """
        a = layer_utils.get_last_valid_elements(tf.constant([
            ['a', 'b', 'c'],
            ['d', 'pad', 'pad']
        ]), batch_size=2, seq_len=tf.constant([3, 1]))
        self.assertAllEqual(a, tf.constant(['c', 'd']))

    def testTileBatch(self):
        """Tests tile_batch() """
        a = layer_utils.tile_batch(tf.constant([['a'], ['b']]), multiplier=2)
        self.assertAllEqual(
            a, tf.constant([
                ['a'], ['a'], ['b'], ['b']
            ])
        )

    def test_get_abstract_tf_function(self):
        class A(ABC):
            @abstractmethod
            @tf.function
            def test_tf_function1(self):
                pass

            @abstractmethod
            def test_function(self):
                pass

            @tf.function
            @abstractmethod
            def test_tf_function2(self):
                pass

        class B(A):
            pass

        abstract_tf_funcs = layer_utils.get_tf_function_names(B)
        self.assertAllEqual(
            abstract_tf_funcs,
            ['test_tf_function1', 'test_tf_function2']
        )


if __name__ == '__main__':
    tf.test.main()
