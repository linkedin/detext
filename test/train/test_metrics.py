import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

from detext.train import metrics
from detext.utils import test_utils

label_padding = tfr.data._PADDING_LABEL


class TestMetrics(tf.test.TestCase):
    """Unit test for metrics."""

    def testNdcg(self):
        """test compute_ndcg()"""
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        labels = tf.placeholder(dtype=tf.float32, shape=[None, None])
        group_size = tf.placeholder(dtype=tf.int32, shape=[None])
        topk = tf.placeholder(dtype=tf.int32, shape=[])
        ndcg = metrics.compute_ndcg(scores, labels, group_size, topk=topk)
        ndcg_tfr = metrics.compute_ndcg_tfr(scores, labels, {}, topk=topk)

        with self.test_session() as sess:
            scores_v = [[3, 2, 3, 0, 1, 2, 2, 1]]
            labels_v = [[0, 0, 0, 0, 0, 0, 0, 0]]
            group_size_v = [8]
            topk_v = 8
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            ndcg_v, ndcg_tfr_v = sess.run([ndcg, ndcg_tfr], feed_dict={
                scores: scores_v,
                labels: labels_v,
                group_size: group_size_v,
                topk: topk_v,
            })
            self.assertAllEqual(ndcg_v[1], ndcg_tfr_v[1], 0)

            scores_v = [[3, 2, 3, 0, 1, 2, 2, 1],
                        [3, 2, 3, 0, 1, 2, 100, -100],
                        [0, 1, 1, 0, 0, 0, 0, 0]]
            labels_v = [[3, 2, 3, 0, 1, 2, 2, 1],
                        [3, 2, 3, 0, 1, 2, label_padding, label_padding],
                        [0, 1, 2, 0, label_padding, label_padding, label_padding, label_padding]]
            group_size_v = [8, 6, 4]
            topk_v = 6
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            ndcg_v, ndcg_tfr_v = sess.run([ndcg, ndcg_tfr], feed_dict={
                scores: scores_v,
                labels: labels_v,
                group_size: group_size_v,
                topk: topk_v
            })
            ndcg2 = test_utils.get_ndcg(scores_v[2][:4], labels_v[2][:4], topk=topk_v)
            ndcg3 = test_utils.compute_ndcg_ps_parity_check(scores_v[2][:4], labels_v[2][:4], topk=topk_v)
            self.assertAlmostEquals((1 + 1 + ndcg2) / 3., ndcg_v[1], places=3)
            self.assertAlmostEqual((1 + 1 + ndcg3) / 3., ndcg_v[1], places=3)

            scores_v = [[6, 5, 4, 3, 2, 1, 0, -1]]
            labels_v = [[3, 2, 3, 0, 1, 2, 3, 2]]
            group_size_v = [8]
            topk_v = 6
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            ndcg_v, ndcg_tfr_v = sess.run([ndcg, ndcg_tfr], feed_dict={
                scores: scores_v,
                labels: labels_v,
                group_size: group_size_v,
                topk: topk_v
            })
            ndcg2 = test_utils.get_ndcg(scores_v[0], labels_v[0], topk=topk_v)
            ndcg3 = test_utils.compute_ndcg_ps_parity_check(scores_v[0], labels_v[0], topk=topk_v)
            ndcg_p2 = test_utils.compute_ndcg_power2(scores_v[0], labels_v[0], topk=topk_v)
            self.assertAlmostEqual(ndcg2, ndcg_v[1], places=3)
            self.assertAlmostEqual(ndcg3, ndcg_v[1], places=3)
            self.assertAlmostEqual(ndcg_p2, ndcg_tfr_v[1], places=3)

            scores_v = [[4, 8, 7, 9, 5, 4.1, 3, 1, 10, 2]]
            labels_v = [[4, 2, 3, 0, 0, 1, 2, 2, 3, 0]]
            group_size_v = [10]
            topk_v = 10
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            ndcg_v, ndcg_tfr_v = sess.run([ndcg, ndcg_tfr], feed_dict={
                scores: scores_v,
                labels: labels_v,
                group_size: group_size_v,
                topk: topk_v
            })
            ndcg2 = test_utils.get_ndcg(scores_v[0], labels_v[0], topk=topk_v)
            ndcg3 = test_utils.compute_ndcg_ps_parity_check(scores_v[0], labels_v[0], topk=topk_v)
            ndcg_p2 = test_utils.compute_ndcg_power2(scores_v[0], labels_v[0], topk=topk_v)
            self.assertAlmostEqual(ndcg2, ndcg_v[1], places=3)
            self.assertAlmostEqual(ndcg3, ndcg_v[1], places=3)
            self.assertAlmostEqual(ndcg_p2, ndcg_tfr_v[1], places=3)

            scores_v = [[4, 8, 7, 9, 5, 4.1, 3, 1, 10, 2]]
            labels_v = [[4, 2, 3, 0, 0, label_padding, label_padding, label_padding, label_padding, label_padding]]
            topk_v = 10
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            ndcg_tfr_v = sess.run(ndcg_tfr, feed_dict={
                scores: scores_v,
                labels: labels_v,
                topk: topk_v
            })
            ndcg_p2 = test_utils.compute_ndcg_power2(
                y_score=scores_v[0][:5],
                y_true=labels_v[0][:5],
                topk=topk_v
            )
            self.assertAlmostEqual(ndcg_p2, ndcg_tfr_v[1], places=3)

    def testComputeRank(self):
        """test compute_rank()"""
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        group_size = tf.placeholder(dtype=tf.int32, shape=[None])
        rank_mat = metrics.compute_rank(scores, group_size)

        with self.test_session() as sess:
            rank_mat_v = sess.run(rank_mat, feed_dict={
                scores: [[1, 2, 3, 4],
                         [5, 1, 2, 6]],
                group_size: [4, 4],
            })
            self.assertAllEqual([[4, 3, 2, 1],
                                 [2, 4, 3, 1]], rank_mat_v)

            rank_mat_v = sess.run(rank_mat, feed_dict={
                scores: [[1, 2, 3, -100],
                         [5, 1, 2, 6]],
                group_size: [3, 2],
            })
            self.assertAllEqual([[3, 2, 1, 4],
                                 [1, 2, 3, 4]], rank_mat_v)

    def testComputePrecision(self):
        """test compute_precision()"""
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        labels = tf.placeholder(dtype=tf.float32, shape=[None, None])
        preat1 = metrics.compute_preat1(scores, labels)
        preat1_tfr = metrics.compute_precision_tfr(scores, labels, {}, 1)

        preat2_tfr = metrics.compute_precision_tfr(scores, labels, {}, 2)

        with self.test_session() as sess:
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            preat1_v, preat1_tfr_v, preat2_tfr_v = sess.run([preat1, preat1_tfr, preat2_tfr], feed_dict={
                scores: [[1, 2, 3, 4],
                         [5, 1, 2, 6]],
                labels: [[0, 0, 1, 1],
                         [1, 0, 0, 0]]
            })
            self.assertAllEqual(0.5, preat1_v[1], preat1_tfr_v[1])
            self.assertAllEqual(0.75, preat2_tfr_v[1])

            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            preat1_v, preat1_tfr_v, preat2_tfr_v = sess.run([preat1, preat1_tfr, preat2_tfr], feed_dict={
                scores: [[1, 2, 3, 4],
                         [5, 1, 2, 6]],
                labels: [[0, 0, 1, 1],
                         [1, 0, 0, label_padding]]
            })
            self.assertAllEqual(0.5, preat1_v[1])
            self.assertAllEqual(1.0, preat1_tfr_v[1])
            self.assertAllEqual(0.75, preat2_tfr_v[1])

    def testMrr(self):
        """test compute_mrr()"""
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        labels = tf.placeholder(dtype=tf.float32, shape=[None, None])
        group_size = tf.placeholder(shape=[None], dtype=tf.int32)
        li_mrr = metrics.compute_mrr(scores, labels, group_size, topk=10)
        mrr = metrics.compute_mrr_tfr(scores, labels, {})

        with self.test_session() as sess:
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            mrr_v, li_mrr_v = sess.run([mrr, li_mrr], feed_dict={
                scores: [[1, 2, 3, 4],
                         [5, 1, 2, 6]],
                labels: [[0, 0, 1, label_padding],
                         [1, 0, 0, 0]],
                group_size: [3, 4]
            })
            self.assertAllEqual(mrr_v[1], 0.75)
            self.assertAllEqual(li_mrr_v[1], 0.75)

    def testAuc(self):
        """Unit test for auc
        Ground truth is obtained by `using roc_auc_score()` from sklearn
        """
        prob = np.array([0.31712287, 0.65969838, 0.52535684, 0.58900034, 0.5647284,
                         0.45878796, 0.92845698, 0.20600347, 0.37837605, 0.17666982,
                         0.6206924, 0.91012624, 0.4409525, 0.94793554, 0.03020039,
                         0.81982164, 0.52474831, 0.14326241, 0.02025829, 0.72717466])
        labels = np.array([1, 1, 0, 0, 1,
                           0, 1, 1, 0, 0,
                           0, 0, 1, 0, 0,
                           0, 1, 1, 1, 1])

        auc = metrics.compute_auc(tf.constant(prob.reshape([-1, 1])), tf.constant(labels.reshape([-1, 1])))
        with tf.Session() as sess:
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            auc_v = sess.run(auc)
            self.assertAlmostEqual(auc_v[1], 0.4, places=3)

        labels = np.ones(labels.shape)
        labels[-1] = 0
        auc = metrics.compute_auc(tf.constant(prob.reshape([-1, 1])), tf.constant(labels.reshape([-1, 1])))
        with tf.Session() as sess:
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            auc_v = sess.run(auc)
            self.assertAlmostEqual(auc_v[1], 0.210526315789, places=3)

        labels = np.zeros(labels.shape)
        labels[-1] = 1
        auc = metrics.compute_auc(tf.constant(prob.reshape([-1, 1])), tf.constant(labels.reshape([-1, 1])))
        with tf.Session() as sess:
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            auc_v = sess.run(auc)
            self.assertAlmostEqual(auc_v[1], 0.789473684210, places=3)

    def testAccuracy(self):
        """
        Unit test for compute_accuracy
        """
        labels = [[0], [3], [2], [2]]
        # Predicted label based on scores: [3, 3, 1, 2]
        scores = [[1.1, 2.1, 3.1, 4.1],
                  [-1, -2, -3, 4],
                  [-2, 2, -1, 0],
                  [1, 2, 5, 3]]
        acc = metrics.compute_accuracy(tf.constant(scores), tf.constant(labels))
        with tf.Session() as sess:
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            acc_v = sess.run(acc)
            self.assertEqual(acc_v[1], 0.5)

    def testConfusionMatrix(self):
        """
        Test compute_confusion_matrix
        """
        labels = [[0], [3], [2], [2], [3], [3]]
        # Predicted label based on scores: [3, 3, 1, 2]
        scores = [[1.1, 2.1, 3.1, 4.1],
                  [-1, -2, -3, 4],
                  [-2, 2, -1, 0],
                  [1, 2, 5, 3],
                  [2, 3, 4, 5],
                  [4, 5, 6, 7]]
        # Expected confusion matrix
        expected_cm = [[0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 3]]
        cm = metrics.compute_confusion_matrix(tf.constant(scores), tf.constant(labels), 4)
        with tf.Session() as sess:
            sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
            cm_v = sess.run(cm)
            self.assertAllEqual(cm_v[1], expected_cm)


if __name__ == "__main__":
    tf.test.main()
