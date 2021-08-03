import math

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

from detext.train.constant import Constant
from detext.train.loss import compute_softmax_loss, compute_regularization_penalty, compute_loss, compute_binary_classification_loss
from detext.utils.parsing_utils import TaskType
from detext.utils.testing import testing_utils


class TestLoss(tf.test.TestCase):
    """Unit test for loss."""
    places = 5
    atol = 10 ** (-places)

    def testComputeLossWithWeight(self):
        """Tests loss correctness with existence of weight

        For inhouse and tfr softmax, we check the correctness and for pairwise & pointwise, we check runnability
        """
        scores_raw = np.array([[1, 2, 3, 4],
                               [1, 1, 1, 1]])
        labels_ranking_raw = np.array([[1, 0, 0, -1],
                                       [0, 1, 0, 0]])
        labels_cls_raw = np.array([3, 2])
        group_size_raw = np.array([3, 4])
        weight_raw = np.array([1, 2])
        var_val = 3.0
        l1_scale = 0.1
        gold_sm_loss = testing_utils.get_softmax_loss(scores_raw, labels_ranking_raw, group_size_raw) / (labels_ranking_raw > 0).sum()
        gold_sm_loss_regularized = np.sum(gold_sm_loss.sum(axis=1) * weight_raw) + np.mean(
            weight_raw) * l1_scale * var_val

        scores = tf.constant(scores_raw, dtype=tf.float32)
        labels_ranking = tf.constant(labels_ranking_raw, dtype=tf.float32)
        labels_cls = tf.constant(labels_cls_raw, dtype=tf.float32)
        weight = tf.constant(weight_raw, dtype=tf.float32)

        # Ranking losses
        tfr_sm_loss = compute_loss(task_type=TaskType.RANKING, use_tfr_loss=True, tfr_loss_fn='softmax_loss',
                                   ltr_loss_fn=None,
                                   tfr_lambda_weights=None, l1=l1_scale, l2=None, scores=scores, labels=labels_ranking,
                                   weight=weight,
                                   trainable_vars=[tf.Variable(var_val, trainable=True)])
        inhouse_sm_loss = compute_loss(task_type=TaskType.RANKING, use_tfr_loss=False, tfr_loss_fn=None,
                                       ltr_loss_fn='softmax', tfr_lambda_weights=None,
                                       l1=l1_scale, l2=None, scores=scores, labels=labels_ranking, weight=weight,
                                       trainable_vars=[tf.Variable(var_val, trainable=True)])
        compute_loss(task_type=TaskType.RANKING, use_tfr_loss=False, ltr_loss_fn='pairwise', tfr_loss_fn=None,
                     tfr_lambda_weights=None,
                     l1=None, l2=None, scores=scores, labels=labels_ranking, weight=weight,
                     trainable_vars=[tf.Variable(var_val, trainable=True)])
        compute_loss(task_type=TaskType.RANKING, use_tfr_loss=False, ltr_loss_fn='pointwise', tfr_loss_fn=None,
                     tfr_lambda_weights=None,
                     l1=None, l2=None, scores=scores, labels=labels_ranking, weight=weight,
                     trainable_vars=[tf.Variable(var_val, trainable=True)])

        self.assertAlmostEqual(tfr_sm_loss, inhouse_sm_loss, places=self.places)
        self.assertAlmostEqual(tfr_sm_loss.numpy(), gold_sm_loss_regularized, places=self.places)

        # Classification losses
        compute_loss(task_type=TaskType.CLASSIFICATION, use_tfr_loss=False, ltr_loss_fn=None, tfr_loss_fn=None,
                     tfr_lambda_weights=None,
                     l1=None, l2=None, scores=scores, labels=labels_cls, weight=weight,
                     trainable_vars=[tf.Variable(var_val, trainable=True)])

    def testSoftmaxLossParityWithTfr(self):
        """Tests consistency of tfr softmax loss and inhouse softmax loss"""
        labels = tf.constant([[1, 0, -1], [0, 0, 1]], dtype=tf.float32)
        scores = tf.constant([[1, 2, 3], [2, 1, 1]], dtype=tf.float32)

        loss_fn = tfr.losses.make_loss_fn('softmax_loss', lambda_weight=None)
        tfr_loss = loss_fn(labels, scores, None)
        loss = compute_softmax_loss(scores, labels)
        loss = tf.reduce_sum(input_tensor=loss) / 2

        self.assertEqual(tfr_loss, loss)

    def testRegularizationPenalty(self):
        """Tests correctness of regularization penalty """

        # Test positive variable
        var = tf.constant(3.0)
        self._testRegularizationPenalty(var, 0.1, 0.2, 0.3, 1.8)

        # Test negative variable
        var = tf.constant(-3.0)
        self._testRegularizationPenalty(var, 0.1, 0.2, 0.3, 1.8)

    def _testRegularizationPenalty(self, var, l1_scale, l2_scale, l1_penalty_truth, l2_penalty_truth):
        """Regularization test for a given variable value"""
        # Get L1 loss
        l1_penalty = compute_regularization_penalty(l1_scale, None, [var])

        # Get L2 loss
        l2_penalty = compute_regularization_penalty(None, l2_scale, [var])

        # Get elastic net loss
        elastic_net_penalty = compute_regularization_penalty(l1_scale, l2_scale, [var])

        self.assertAlmostEqual(l1_penalty.numpy(), l1_penalty_truth, places=self.places)
        self.assertAlmostEqual(l2_penalty.numpy(), l2_penalty_truth, places=self.places)
        self.assertAlmostEqual(elastic_net_penalty.numpy(), l1_penalty_truth + l2_penalty_truth, places=self.places)

    def _testSoftmaxInExtremeCase(self):
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
        loss = compute_softmax_loss(scores, labels)

        max_scores = tf.reduce_max(input_tensor=scores, axis=-1, keepdims=True)
        mask = tf.sequence_mask(group_size, maxlen=tf.shape(input=scores)[1], dtype=tf.float32)
        scores = scores - max_scores
        exp_score_withmask = tf.exp(scores) * tf.cast(mask, dtype=tf.float32)
        logsumexp = tf.math.log(tf.reduce_sum(input_tensor=exp_score_withmask, axis=-1)) + tf.squeeze(max_scores, -1)
        loss_ori = labels * (-scores + tf.expand_dims(logsumexp, axis=1))

        self.assertTrue(not math.isnan(loss.numpy()))
        self.assertTrue(not math.isnan(tfr_loss.numpy()))
        self.assertTrue(math.isinf(logsumexp.numpy()))
        self.assertTrue(math.isnan(loss_ori.numpy().sum()))

    def testComputeBinaryClassificationLoss(self):
        scores = tf.constant([0.0, 20.0], dtype=tf.dtypes.float32)
        labels = tf.constant([1, 0], dtype=tf.dtypes.float32)
        weight = tf.constant([1.0, 1.0], dtype=tf.dtypes.float32)
        compute_binary_classification_loss(scores=scores, labels=labels, weight=weight)

    def testComputeSoftmaxLoss(self):
        scores_lst = [
            tf.constant([[0, 1]], dtype=tf.dtypes.float32),
            tf.constant([[2, 1, 3, 0],
                         [5, 2, 7, 9]], dtype=tf.dtypes.float32),
        ]
        labels_lst = [
            tf.constant([[0, 1]], dtype=tf.dtypes.float32),
            tf.constant([[1, 0, Constant()._LABEL_PADDING, Constant()._LABEL_PADDING],
                         [1, 3, 2, Constant()._LABEL_PADDING]], dtype=tf.dtypes.float32),
        ]
        group_size_lst = [
            tf.constant([2]),
            tf.constant([2, 3]),
        ]
        self.assertTrue(len(scores_lst) == len(labels_lst) == len(group_size_lst))
        for scores, labels, group_size in zip(scores_lst, labels_lst, group_size_lst):
            self._testComputeSoftmaxLoss(scores, labels, group_size)

    def _testComputeSoftmaxLoss(self, scores, labels, group_size):
        """Tests softmax loss given input """
        loss = compute_softmax_loss(scores, labels)
        self.assertAllClose(testing_utils.get_softmax_loss(scores, labels, group_size), loss, atol=self.atol)


if __name__ == "__main__":
    tf.test.main()
