import tensorflow as tf

from detext.layers import interaction_layer
from detext.utils.parsing_utils import InputFtrType


class ScoringLayer(tf.keras.layers.Layer):
    """Scoring layer that performs linear projection from interaction features to scalar scores"""

    def __init__(self, task_ids, num_classes):
        super(ScoringLayer, self).__init__()
        if task_ids is None:
            task_ids = [0]

        self._task_ids = task_ids
        self._num_classes = num_classes
        self.final_projections = self.create_final_projections(task_ids, num_classes)

    def compute_final_scores(self, ftrs_to_score, task_id):
        """Returns final scores given interaction outputs wrt the given task"""
        if len(self._task_ids) <= 1 or task_id is None:
            return self.final_projections[0](ftrs_to_score[self.get_scoring_ftrs_key(0)])

        # Multitask
        data_shape = tf.shape(ftrs_to_score[self.get_scoring_ftrs_key(0)])
        batch_size = data_shape[0]
        max_group_size = data_shape[1]

        score_shape = [batch_size, tf.maximum(max_group_size, 1), tf.maximum(1, self._num_classes)]
        scores = tf.zeros(shape=score_shape, dtype="float32")
        for i, ith_task_id in enumerate(self._task_ids):
            task_score = self.final_projections[i](ftrs_to_score[self.get_scoring_ftrs_key(i)])
            task_mask = tf.cast(tf.equal(task_id, int(ith_task_id)), dtype=tf.float32)
            # Broadcast task_mask for compatible tensor shape with scores tensor
            task_mask = tf.transpose(a=tf.broadcast_to(task_mask, score_shape[::-1]))
            scores += task_mask * task_score
        return scores

    @staticmethod
    def create_final_projections(task_ids, num_classes):
        """Returns a list of final projection layers for given task_ids """
        final_projections = []
        # When task_ids is None, treat it as a special case of multitask learning when there's only one task
        if task_ids is None:
            task_ids = [0]
        # Set up layers for each task
        for task_id in task_ids:
            final_projections.append(tf.keras.layers.Dense(num_classes, name=f"task_{task_id}_final_projection"))
        return final_projections

    @staticmethod
    def get_scoring_ftrs_key(i):
        return interaction_layer.InteractionLayer.get_interaction_ftrs_key(i)

    def get_ftrs_to_score(self, inputs):
        return interaction_layer.InteractionLayer.get_interaction_ftrs(inputs, self._task_ids)

    def call(self, inputs, **kwargs):
        """ Projects features linearly to scores (scalar)

        :param inputs: Map {
            InternalFtrType.FTRS_TO_SCORE: Tensor(dtype=float32, shape=[batch_size, list_size, num_features])
            InternalFtrType.TASK_ID: Tensor(dtype=int32, shape=[batch_size])
        }
        :param kwargs:
        :return: scores. Tensor(dtype=float32, shape=[batch_size, list_size])
        """
        ftrs_to_score = self.get_ftrs_to_score(inputs)
        task_id = inputs.get(InputFtrType.TASK_ID_COLUMN_NAME, None)

        return self.compute_final_scores(ftrs_to_score, task_id)
