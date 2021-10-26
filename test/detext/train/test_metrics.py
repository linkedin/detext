import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

from detext.train import metrics
from detext.utils.parsing_utils import TaskType
from detext.utils.testing import testing_utils

label_padding = tfr.data._PADDING_LABEL


class TestMetrics(tf.test.TestCase):
    """Unit test for metrics."""
    places = 3

    def testNdcg(self):
        """Test compute_ndcg()"""
        scores_lst = [
            tf.constant([[6, 5, 4, 3, 2, 1, 0, -1]], dtype=tf.dtypes.float32),
            tf.constant([[4, 8, 7, 9, 5, 4.1, 3, 1, 10, 2]], dtype=tf.dtypes.float32),
            tf.constant([[4, 8, 7, 9, 5, 4.1, 3, 1, 10, 2]], dtype=tf.dtypes.float32)
        ]
        labels_lst = [
            tf.constant([[3, 2, 3, 0, 1, 2, 3, 2]], dtype=tf.dtypes.float32),
            tf.constant([[4, 2, 3, 0, 0, 1, 2, 2, 3, 0]], dtype=tf.dtypes.float32),
            tf.constant([[4, 2, 3, 0, 0, label_padding, label_padding, label_padding, label_padding, label_padding]],
                        dtype=tf.dtypes.float32)
        ]
        topk_lst = [
            6, 10, 10
        ]
        scores_for_test_utils_lst = [
            np.array([6, 5, 4, 3, 2, 1, 0, -1]),
            np.array([4, 8, 7, 9, 5, 4.1, 3, 1, 10, 2]),
            np.array([4, 8, 7, 9, 5]),
        ]
        labels_for_test_utils_lst = [
            np.array([3, 2, 3, 0, 1, 2, 3, 2]),
            np.array([4, 2, 3, 0, 0, 1, 2, 2, 3, 0]),
            np.array([4, 2, 3, 0, 0])
        ]
        self.assertTrue(len(scores_lst) == len(labels_lst) == len(topk_lst) == len(
            scores_for_test_utils_lst) == len(labels_for_test_utils_lst))
        for scores, labels, scores_for_test_utils, labels_for_test_utils, topk in zip(scores_lst, labels_lst,
                                                                                      scores_for_test_utils_lst,
                                                                                      labels_for_test_utils_lst,
                                                                                      topk_lst):
            self._testNdcg(scores, labels, scores_for_test_utils, labels_for_test_utils, topk)

    def _testNdcg(self, scores, labels, scores_for_test_utils, labels_for_test_utils, topk):
        """Test compute_ndcg() given input data """
        metric_name = f'ndcg@{topk}'
        metric = metrics.get_metric_fn(metric_name, task_type=TaskType.RANKING, num_classes=None)()
        metric.update_state(labels, scores)
        ndcg_tfr = metric.result().numpy()
        ndcg_p2 = testing_utils.compute_ndcg_power2(scores_for_test_utils, labels_for_test_utils, topk=topk)
        self.assertEqual(metric.name, metric_name)
        self.assertAlmostEqual(ndcg_p2, ndcg_tfr, places=self.places)

    def testComputePrecision(self):
        """ Test compute_precision() """
        scores_lst = [
            tf.constant([[1, 2, 3, 4],
                         [5, 1, 2, 6]], dtype=tf.dtypes.float32),
            tf.constant([[1, 2, 3, 4],
                         [5, 1, 2, 6]], dtype=tf.dtypes.float32)
        ]
        labels_lst = [
            tf.constant([[0, 0, 1, 1],
                         [1, 0, 0, 0]], dtype=tf.dtypes.float32),
            tf.constant([[0, 0, 1, 1],
                         [1, 0, 0, label_padding]], dtype=tf.dtypes.float32)
        ]
        expected_precision_at_1_lst = [
            0.5, 1.0
        ]
        expected_precision_at_2_lst = [
            0.75, 0.75
        ]
        self.assertTrue(
            len(scores_lst) == len(labels_lst) == len(expected_precision_at_1_lst) == len(expected_precision_at_2_lst))
        for scores, labels, expected_precision_at_1, expected_precision_at_2 in zip(
                scores_lst, labels_lst, expected_precision_at_1_lst, expected_precision_at_2_lst
        ):
            self._testComputePrecision(scores, labels, expected_precision_at_1, expected_precision_at_2)

    def _testComputePrecision(self, scores, labels, expected_precision_at_1, expected_precision_at_2):
        """ Test compute_precision() given input data """
        metric_name_1 = 'precision@1'
        metric_at_1 = metrics.get_metric_fn(metric_name_1, task_type=TaskType.RANKING, num_classes=None)()
        metric_at_1.update_state(labels, scores)
        preat1_tfr = metric_at_1.result().numpy()
        self.assertEqual(metric_at_1.name, metric_name_1)

        metric_name_2 = 'precision@2'
        metric_at_2 = metrics.get_metric_fn(metric_name_2, task_type=TaskType.RANKING, num_classes=None)()
        metric_at_2.update_state(labels, scores)
        preat2_tfr = metric_at_2.result().numpy()
        self.assertEqual(metric_at_2.name, metric_name_2)

        self.assertAllEqual(expected_precision_at_1, preat1_tfr)
        self.assertAllEqual(expected_precision_at_2, preat2_tfr)

    def testMrr(self):
        """ Test compute_mrr() """
        scores = [[1, 2, 3, 4],
                  [5, 1, 2, 6]]
        labels = [[0, 0, 1, label_padding],
                  [1, 0, 0, 0]]
        topk = 4

        metric_name = f'mrr@{topk}'
        metric = metrics.get_metric_fn(metric_name, task_type=TaskType.RANKING, num_classes=None)()
        metric.update_state(labels, scores)
        mrr = metric.result().numpy()
        self.assertEqual(metric.name, metric_name)
        self.assertAllEqual(mrr, 0.75)

    def testAuc(self):
        """ Unit test for auc

        Ground truth is obtained by `using roc_auc_score()` from sklearn
        """

        labels_lst = [np.array([1, 1, 0, 0, 1,
                                0, 1, 1, 0, 0,
                                0, 0, 1, 0, 0,
                                0, 1, 1, 1, 1]),
                      np.array([1] * 19 + [0]),
                      np.array([0] * 19 + [1]),
                      ]

        scores_lst = [np.array([0.31712287, 0.65969838, 0.52535684, 0.58900034, 0.5647284,
                                0.45878796, 0.92845698, 0.20600347, 0.37837605, 0.17666982,
                                0.6206924, 0.91012624, 0.4409525, 0.94793554, 0.03020039,
                                0.81982164, 0.52474831, 0.14326241, 0.02025829, 0.72717466])] * len(labels_lst)
        expected_auc_lst = [0.4, 0.210526315789, 0.789473684210, ]
        self.assertTrue(len(labels_lst) == len(scores_lst) == len(expected_auc_lst))
        for prob, labels, expected_auc in zip(scores_lst, labels_lst, expected_auc_lst):
            self._testAuc(prob, labels, expected_auc)

    def _testAuc(self, prob, labels, expected_auc):
        """ Unit test for auc

        Ground truth is obtained by `using roc_auc_score()` from sklearn
        """
        for task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.RANKING]:
            metric_name = 'auc'
            metric = metrics.get_metric_fn(metric_name, task_type=task_type, num_classes=1)()
            metric.reset_states()
            metric.update_state(tf.constant(labels.reshape([-1, 1])), tf.constant(prob.reshape([-1, 1])))
            auc = metric.result().numpy()
            self.assertEqual(metric.name, metric_name)
            self.assertAlmostEqual(auc, expected_auc, places=self.places)

    def testAccuracy(self):
        """ Unit test for compute_accuracy """
        labels = [0, 3, 2, 2.]
        # Predicted label based on scores: [3, 3, 1, 2]
        scores = [[1.1, 2.1, 3.1, 4.1],
                  [-1, -2, -3, 4],
                  [-2, 2, -1, 0],
                  [1, 2, 5, 3]]
        metric_name = 'accuracy'
        # Classification
        metric = metrics.get_metric_fn(metric_name, task_type=TaskType.CLASSIFICATION, num_classes=4)()
        metric.reset_states()
        metric.update_state(tf.constant(labels), tf.constant(scores))
        acc = metric.result().numpy()
        self.assertEqual(metric.name, metric_name)
        self.assertEqual(acc, 0.5)

        # Binary classification
        labels = [0, 1, 1, 0.]
        scores = [1, -1, 1, 1.]
        metric = metrics.get_metric_fn(metric_name, task_type=TaskType.BINARY_CLASSIFICATION, num_classes=1)()
        metric.reset_states()
        metric.update_state(tf.constant(labels), tf.constant(scores))
        acc = metric.result().numpy()
        self.assertEqual(metric.name, metric_name)
        self.assertEqual(acc, 0.25)

    def testConfusionMatrix(self):
        """ Test compute_confusion_matrix """
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
        metric_name = 'confusion_matrix'
        metric = metrics.get_metric_fn(metric_name, num_classes=4, task_type=TaskType.CLASSIFICATION)()
        metric.update_state(tf.constant(labels), tf.constant(scores))
        cm = metric.result().numpy()
        self.assertEqual(metric.name, metric_name)
        self.assertAllEqual(cm, expected_cm)


if __name__ == "__main__":
    tf.test.main()
