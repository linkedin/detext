import tensorflow as tf

from smart_compose.train.losses import compute_text_generation_loss, compute_regularization_penalty, compute_loss
from smart_compose.utils.testing.test_case import TestCase


class TestLoss(TestCase):
    """Unit test for losses.py"""
    places = 5
    atol = 10 ** (-places)

    def testComputeLoss(self):
        """Tests compute_loss() """
        labels = tf.constant([[1, 2],
                              [1, 2]], dtype=tf.dtypes.float32)
        logits = tf.constant([[[0, 1, 0], [0, 0, 1]],
                              [[0, 1, 0], [0, 0, 1]]], dtype=tf.dtypes.float32)
        lengths = tf.constant([2, 2], dtype=tf.dtypes.int32)
        text_generation_loss = compute_text_generation_loss(logits=logits, labels=labels, lengths=lengths)

        l1_scale = l2_scale = 0.1
        var = tf.constant(3.0)
        regularization_penalty = compute_regularization_penalty(l1_scale, l2_scale, [var])  # Get elastic net loss

        self.assertEqual(text_generation_loss + regularization_penalty, compute_loss(l1=l1_scale, l2=l2_scale,
                                                                                     logits=logits, labels=labels, lengths=lengths,
                                                                                     trainable_vars=[var]))

    def testComputeTextGenerationLoss(self):
        """Tests compute_text_generation_loss()"""
        # Loss of correct prediction must be smaller than that of incorrect prediction
        labels = tf.constant([[1, 2],
                              [1, 2]], dtype=tf.dtypes.float32)
        logits = tf.constant([[[0, 1, 0], [0, 0, 1]],
                              [[0, 1, 0], [0, 0, 1]]], dtype=tf.dtypes.float32)
        lengths = tf.constant([2, 2], dtype=tf.dtypes.int32)
        loss_of_correct_prediction = compute_text_generation_loss(logits=logits, labels=labels, lengths=lengths)

        labels = tf.constant([[1, 2],
                              [1, 2]], dtype=tf.dtypes.float32)
        logits = tf.constant([[[1, 0, 0], [1, 0, 0]],
                              [[1, 0, 0], [1, 0, 0]]], dtype=tf.dtypes.float32)
        lengths = tf.constant([2, 2], dtype=tf.dtypes.int32)
        loss_of_incorrect_prediction = compute_text_generation_loss(logits=logits, labels=labels, lengths=lengths)
        self.assertAllGreater(loss_of_incorrect_prediction, loss_of_correct_prediction)

        # Verify effectiveness of length: when length = 0, loss must be 0
        labels = tf.constant([[1, 2],
                              [1, 2]], dtype=tf.dtypes.float32)
        logits = tf.constant([[[0, 1, 0], [0, 0, 1]],
                              [[0, 1, 0], [0, 0, 1]]], dtype=tf.dtypes.float32)
        lengths = tf.constant([0, 0], dtype=tf.dtypes.int32)
        zero_length_loss = compute_text_generation_loss(logits=logits, labels=labels, lengths=lengths)
        self.assertEqual(zero_length_loss, 0)

    def testRegularizationPenalty(self):
        """Tests correctness of regularization penalty """
        # Test positive variable
        var = tf.constant(3.0)
        self._testRegularizationPenalty(var, 0.1, 0.2, 0.3, 1.8)

        # Test negative variable
        var = tf.constant(-3.0)
        self._testRegularizationPenalty(var, 0.1, 0.2, 0.3, 1.8)

    def _testRegularizationPenalty(self, var, l1_scale, l2_scale, l1_penalty_truth, l2_penalty_truth):
        """Tests regularization for a given variable value"""
        l1_penalty = compute_regularization_penalty(l1_scale, None, [var])  # Get L1 loss
        l2_penalty = compute_regularization_penalty(None, l2_scale, [var])  # Get L2 loss
        elastic_net_penalty = compute_regularization_penalty(l1_scale, l2_scale, [var])  # Get elastic net loss

        self.assertAlmostEqual(l1_penalty.numpy(), l1_penalty_truth, places=self.places)
        self.assertAlmostEqual(l2_penalty.numpy(), l2_penalty_truth, places=self.places)
        self.assertAlmostEqual(elastic_net_penalty.numpy(), l1_penalty_truth + l2_penalty_truth, places=self.places)


if __name__ == '__main__':
    tf.test.main()
