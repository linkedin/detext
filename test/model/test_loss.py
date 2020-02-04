import math

import tensorflow as tf
import tensorflow_ranking as tfr

from detext.model.softmax_loss import compute_softmax_loss


class TestLoss(tf.test.TestCase):
    """Unit test for loss."""

    def testLoss(self):
        group_size = tf.constant([2, 3], dtype=tf.int32)
        labels = tf.constant([[1, 0, -1], [0, 0, 1]], dtype=tf.float32)
        scores = tf.constant([[1, 2, 3], [2, 1, 1]], dtype=tf.float32)

        loss_fn = tfr.losses.make_loss_fn('softmax_loss', lambda_weight=None)
        tfr_loss = loss_fn(labels, scores, None)
        loss = compute_softmax_loss(scores, labels, group_size)
        loss = tf.reduce_sum(loss) / 2

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            loss1, loss2 = sess.run([tfr_loss, loss])
            self.assertEqual(loss1, loss2)

    def testSoftmaxInExtremeCase(self):
        labels = [[1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
        scores = [[-67.9612198, -70.6147385, 19.9753132, 19.9753132, 19.9753132, 19.9753132, 19.9753132, 19.9753132,
                   19.9753132, 19.9753132, 19.9753132, 19.9753132, 19.9753132, 19.9753132, 19.9753132, 19.9753132,
                   19.9753132, 19.9753132, 19.9753132, 19.9753132]]

        group_size = tf.constant([sum(map(lambda q: q != -1, x)) for x in labels], dtype=tf.int32)
        labels = tf.constant(labels, dtype=tf.float32)
        scores = tf.constant(scores, dtype=tf.float32)

        loss_fn = tfr.losses.make_loss_fn('softmax_loss', lambda_weight=None)
        tfr_loss = loss_fn(labels, scores, None)
        loss = compute_softmax_loss(scores, labels, group_size)
        loss = tf.reduce_sum(loss) / 2

        max_scores = tf.reduce_max(scores, axis=-1, keepdims=True)
        mask = tf.sequence_mask(group_size, maxlen=tf.shape(scores)[1], dtype=tf.float32)
        scores = scores - max_scores
        exp_score_withmask = tf.exp(scores) * tf.cast(mask, dtype=tf.float32)
        logsumexp = tf.log(tf.reduce_sum(exp_score_withmask, axis=-1)) + tf.squeeze(max_scores, -1)
        loss_ori = labels * (-scores + tf.expand_dims(logsumexp, axis=1))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            loss1, loss2, logsumexp1, loss_ori1 = sess.run([tfr_loss, loss, logsumexp, loss_ori])
            self.assertTrue(not math.isnan(loss2))
            self.assertTrue(math.isinf(logsumexp1))
            self.assertTrue(math.isnan(loss_ori1.sum()))


if __name__ == "__main__":
    tf.test.main()
