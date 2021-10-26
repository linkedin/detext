from itertools import product

import tensorflow as tf

from detext.layers import scoring_layer
from detext.utils.parsing_utils import InputFtrType
from detext.utils.testing.data_setup import DataSetup


class TestEmbeddingInteractionLayer(tf.test.TestCase, DataSetup):
    """Unit test for scoring_layer.py"""
    batch_size = 3
    list_size = 10
    interaction_ftr_size = 5

    def testScoringLayer(self):
        """Tests ScoringLayer"""
        task_ids_list = [[0], [213, 3213, 4]]
        num_classes_list = [1, 20]
        for task_ids, num_classes in product(task_ids_list, num_classes_list):
            self._testScoringLayer(task_ids, num_classes)

    def _testScoringLayer(self, task_ids, num_classes):
        """Tests ScoringLayer under given settings"""
        inputs = {
            InputFtrType.TASK_ID_COLUMN_NAME: tf.constant(0, dtype=tf.dtypes.int32),
            **{scoring_layer.ScoringLayer.get_scoring_ftrs_key(i): tf.random.uniform([self.batch_size, self.list_size, self.interaction_ftr_size])
               for i in range(len(task_ids))}
        }

        layer = scoring_layer.ScoringLayer(task_ids, num_classes)
        outputs = layer(inputs)
        self.assertAllEqual(tf.shape(outputs), [self.batch_size, self.list_size, num_classes])


if __name__ == '__main__':
    tf.test.main()
