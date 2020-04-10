import math

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

from detext.train.loss import compute_softmax_loss, compute_regularization_penalty
from detext.train.train import compute_loss
from detext.utils import test_utils


class TestLoss(tf.test.TestCase):
    """Unit test for loss."""

    def testComputeLossWithWeight(self):
        """Tests loss correctness with existence of weight

        For inhouse and tfr softmax, we check the correctness and for pairwise & pointwise, we check runnability
        """
        scores_raw = np.array([[1, 2, 3, 4],
                               [1, 1, 1, 1]])
        labels_raw = np.array([[1, 0, 0, -1],
                               [0, 1, 0, 0]])
        group_size_raw = np.array([3, 4])
        weight_raw = np.array([1, 2])
        var_val = 3
        l1_scale = 0.1
        gold_sm_loss = test_utils.get_softmax_loss(scores_raw, labels_raw, group_size_raw) / (labels_raw > 0).sum()
        gold_sm_loss_regularized = np.sum(gold_sm_loss.sum(axis=1) * weight_raw) + np.mean(
            weight_raw) * l1_scale * var_val

        tfr_sm_hparams = tf.contrib.training.HParams(num_classes=0, use_tfr_loss=True, tfr_loss_fn='softmax_loss',
                                                     tfr_lambda_weights=None, l1=l1_scale, l2=None)
        inhouse_sm_hparams = tf.contrib.training.HParams(num_classes=0, use_tfr_loss=False, ltr_loss_fn='softmax',
                                                         l1=l1_scale, l2=None)
        inhouse_pair_hparams = tf.contrib.training.HParams(num_classes=0, use_tfr_loss=False, ltr_loss_fn='pairwise',
                                                           l1=None, l2=None)
        inhouse_pt_hparams = tf.contrib.training.HParams(num_classes=0, use_tfr_loss=False, ltr_loss_fn='pointwise',
                                                         l1=None, l2=None)
        with tf.Graph().as_default() as g:
            tf.Variable(var_val, dtype=tf.float32, trainable=True)
            scores = tf.constant(scores_raw, dtype=tf.float32)
            labels = tf.constant(labels_raw, dtype=tf.float32)
            group_size = tf.constant(group_size_raw)
            weight = tf.constant(weight_raw, dtype=tf.float32)
            tfr_sm_loss = compute_loss(tfr_sm_hparams, scores, labels, group_size, weight)
            inhouse_sm_loss = compute_loss(inhouse_sm_hparams, scores, labels, group_size, weight)
            inhouse_pair_loss = compute_loss(inhouse_pair_hparams, scores, labels, group_size, weight)
            inhouse_pt_loss = compute_loss(inhouse_pt_hparams, scores, labels, group_size, weight)
            with tf.Session(graph=g) as sess:
                sess.run([tf.global_variables_initializer()])
                self.assertAlmostEqual(tfr_sm_loss.eval(), inhouse_sm_loss.eval())
                self.assertAlmostEqual(tfr_sm_loss.eval(), gold_sm_loss_regularized, places=5)
                inhouse_pair_loss.eval()
                inhouse_pt_loss.eval()

    def testSoftmaxLossParityWithTfr(self):
        """Tests consistency of tfr softmax loss and inhouse softmax loss"""
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

    def testRegularizationPenalty(self):
        """Tests correctness of regularization penalty """

        def _testRegularizationPenalty(var, l1_scale, l2_scale, l1_penalty_truth, l2_penalty_truth):
            """Regularization test for a given variable value"""
            with tf.Graph().as_default() as g:
                tf.Variable(var, trainable=True)  # Register variable in the graph

                # Get L1 loss
                l1_hparams = tf.contrib.training.HParams(l1=l1_scale, l2=None)
                l1_penalty = compute_regularization_penalty(l1_hparams)

                # Get L2 loss
                l2_hparams = tf.contrib.training.HParams(l1=None, l2=l2_scale)
                l2_penalty = compute_regularization_penalty(l2_hparams)

                # Get elastic net loss
                elastic_net_hparams = tf.contrib.training.HParams(l1=l1_scale, l2=l2_scale)
                elastic_net_penalty = compute_regularization_penalty(elastic_net_hparams)
                with tf.Session(graph=g) as sess:
                    sess.run([tf.global_variables_initializer()])
                    self.assertAlmostEqual(l1_penalty.eval(), l1_penalty_truth)
                    self.assertAlmostEqual(l2_penalty.eval(), l2_penalty_truth)
                    self.assertAlmostEqual(elastic_net_penalty.eval(), l1_penalty_truth + l2_penalty_truth)

        # Test positive value
        var = 3.0
        _testRegularizationPenalty(var, 0.1, 0.2, 0.3, 0.9)

        # Test negative value
        var = -3.0
        _testRegularizationPenalty(var, 0.1, 0.2, 0.3, 0.9)

    def testSoftmaxInExtremeCase(self):
        """Tests softmax loss correctness when difference between padding score and valid score is large """
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

    def testComputeSoftmaxLoss(self):
        """Tests softmax loss"""
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        labels = tf.placeholder(dtype=tf.float32, shape=[None, None])
        group_size = tf.placeholder(dtype=tf.int32, shape=[None])
        loss = compute_softmax_loss(scores, labels, group_size)

        with self.test_session() as sess:
            # a simple case
            loss_v = sess.run(loss, feed_dict={
                scores: [[0, 1]],
                labels: [[0, 1]],
                group_size: [2],
            })
            self.assertAllClose(test_utils.get_softmax_loss([[0, 1]], [[0, 1]], [2]), loss_v)

            # a more complicated case
            loss_v = sess.run(loss, feed_dict={
                scores: [[2, 1, 3, 0],
                         [5, 2, 7, 9]],
                labels: [[1, 0, 0, 2],
                         [1, 3, 2, 0]],
                group_size: [2, 3],
            })
            loss_v2 = test_utils.get_softmax_loss(
                [[2, 1, 3, 0],
                 [5, 2, 7, 9]],
                [[1, 0, 0, 2],
                 [1, 3, 2, 0]],
                [2, 3]
            )
            self.assertAllClose(loss_v2, loss_v)


if __name__ == "__main__":
    tf.test.main()
