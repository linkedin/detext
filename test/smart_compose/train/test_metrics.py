import tensorflow as tf

from smart_compose.train.metrics import NegativePerplexity
from smart_compose.utils.testing.test_case import TestCase


class TestMetrics(TestCase):
    """Unit test for metrics.py"""

    def testPerplexity(self):
        """Tests class Perplexity"""
        metric = NegativePerplexity()
        metric.reset_states()

        initial_value = metric.result()
        self.assertEqual(initial_value, -1)

        labels = tf.constant([[1, 2],
                              [1, 2]], dtype=tf.dtypes.int32)
        logits = tf.constant([[[0, 1, 0], [0, 0, 1]],
                              [[0, 1, 0], [0, 0, 1]]], dtype=tf.dtypes.float32)
        lengths = tf.constant([2, 2], dtype=tf.dtypes.int32)
        metric.update_state(labels=labels, logits=logits, lengths=lengths)
        updated_value = metric.result()
        perplexity_val = updated_value.numpy()

        metric.update_state(labels=labels, logits=logits, lengths=lengths)
        updated_value = metric.result()
        # Perplexity should be the same if inputs of the two updates are the same
        self.assertEqual(updated_value, perplexity_val)

        metric.update_state(labels=labels - 1, logits=logits, lengths=lengths)
        updated_value = metric.result()
        # Perplexity should change if once a different input is given
        self.assertNotEqual(updated_value, perplexity_val)


if __name__ == '__main__':
    tf.test.main()
