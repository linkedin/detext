from functools import partial

import tensorflow as tf

from detext.layers.multi_layer_perceptron import MultiLayerPerceptron
from detext.utils.parsing_utils import InternalFtrType


def _get_num_sim_ftrs_for_inner(num_doc_fields, num_user_fields, *args):
    return num_doc_fields * num_user_fields


def _get_num_sim_ftrs_for_concat(num_doc_fields, num_user_fields, num_deep):
    return (num_doc_fields + num_user_fields) * num_deep


def _get_num_sim_ftrs_for_hadamard(num_doc_fields, num_user_fields, num_deep):
    return num_doc_fields * num_user_fields * num_deep


def _get_num_sim_ftrs_for_diff(num_doc_fields, num_user_fields, num_deep):
    return num_doc_fields * num_user_fields * num_deep


NAME_TO_NUM_SIM = {'inner': _get_num_sim_ftrs_for_inner,
                   'concat': _get_num_sim_ftrs_for_concat,
                   'hadamard': _get_num_sim_ftrs_for_hadamard,
                   'diff': _get_num_sim_ftrs_for_diff}


def compute_num_sim_ftrs(sim_func_names, num_doc_fields, num_user_fields, num_deep):
    """Computes the number of similarity features

    :param num_doc_fields Number of document fields
    :param num_user_fields Number of user fields
    :param num_deep Number of deep features
    :param sim_func_names Similarity function names
    """
    num_sim_ftrs = 0
    for name in sim_func_names:
        num_sim_ftrs += NAME_TO_NUM_SIM[name](num_doc_fields, num_user_fields, num_deep)
    return num_sim_ftrs


def compute_elem_diff_for_user_doc(doc_ftrs, user_ftrs, num_doc_fields, num_user_fields, num_deep):
    """Computes Elementwise difference between user_field_ftrs and doc_ftrs

    :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param user_ftrs Shape=[batch_size, num_user_fields, num_deep]
    :param num_doc_fields Number of document fields
    :param num_user_fields Number of user fields
    :param num_deep Number of deep features
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features (the last dimension
    of sim_ftrs)
    """

    def compute_diff(user_field_ftrs, doc_ftrs):
        """ Computes Elementwise diff between user_field_ftrs and doc_ftrs

        :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
        :param user_field_ftrs Shape=[batch_size, num_deep]
        """
        user_field_ftrs = tf.expand_dims(tf.expand_dims(user_field_ftrs, axis=1), axis=1)  # [batch_size, 1, 1, num_deep]
        elementwise_diff = user_field_ftrs - doc_ftrs  # [batch_size, group_size, num_doc_fields, num_deep]
        return elementwise_diff  # [batch_size, group_size, num_doc_fields, num_deep]

    batch_size = tf.shape(input=doc_ftrs)[0]
    max_group_size = tf.shape(input=doc_ftrs)[1]

    # Shape=[num_user_fields, batch_size, group_size, num_doc_fields, num_deep]
    sim_ftrs = tf.map_fn(partial(compute_diff, doc_ftrs=doc_ftrs), tf.transpose(a=user_ftrs, perm=[1, 0, 2]))
    # Shape=[batch_size, group_size, num_doc_fields, num_user_fields, num_deep]
    sim_ftrs = tf.transpose(a=sim_ftrs, perm=[1, 2, 3, 4, 0])

    num_sim_ftrs = NAME_TO_NUM_SIM['diff'](num_doc_fields, num_user_fields, num_deep)
    # Shape=[batch_size, group_size, num_sim_ftrs]
    sim_ftrs = tf.reshape(sim_ftrs, [batch_size, max_group_size, num_sim_ftrs])
    return sim_ftrs


def compute_hadamard_prod_for_user_doc(doc_ftrs, user_ftrs, num_doc_fields, num_user_fields, num_deep):
    """Computes Hadamard product between user_field_ftrs and doc_ftrs

    :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param user_ftrs Shape=[batch_size, num_user_fields, num_deep]
    :param num_doc_fields Number of document fields
    :param num_user_fields Number of user fields
    :param num_deep Number of deep features
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features (the last dimension
    of sim_ftrs)
    """

    def compute_hadamard(user_field_ftrs, doc_ftrs):
        """ Computes Hadamard product between user_field_ftrs and doc_ftrs

        :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
        :param user_field_ftrs Shape=[batch_size, num_deep]
        """
        user_field_ftrs = tf.expand_dims(tf.expand_dims(user_field_ftrs, axis=1), axis=1)  # [batch_size, 1, 1, num_deep]
        hadamard_prod = user_field_ftrs * doc_ftrs  # [batch_size, group_size, num_doc_fields, num_deep]
        return hadamard_prod  # [batch_size, group_size, num_doc_fields, num_deep]

    batch_size = tf.shape(input=doc_ftrs)[0]
    max_group_size = tf.shape(input=doc_ftrs)[1]

    # Shape=[num_user_fields, batch_size, group_size, num_doc_fields, num_deep]
    sim_ftrs = tf.map_fn(partial(compute_hadamard, doc_ftrs=doc_ftrs), tf.transpose(a=user_ftrs, perm=[1, 0, 2]))
    # Shape=[batch_size, group_size, num_doc_fields, num_user_fields, num_deep]
    sim_ftrs = tf.transpose(a=sim_ftrs, perm=[1, 2, 3, 4, 0])

    num_sim_ftrs = NAME_TO_NUM_SIM['hadamard'](num_doc_fields, num_user_fields, num_deep)
    # Shape=[batch_size, group_size, num_sim_ftrs]
    sim_ftrs = tf.reshape(sim_ftrs, [batch_size, max_group_size, num_sim_ftrs])
    return sim_ftrs


def compute_concat_ftrs_for_user_doc(doc_ftrs, user_ftrs, num_doc_fields, num_user_fields, num_deep):
    """Concatenates doc_ftrs and user_ftrs

    :param doc_ftrs Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param user_ftrs Shape=[batch_size, num_user_fields, num_deep]
    :param num_doc_fields Number of document fields
    :param num_user_fields Number of user fields
    :param num_deep Number of deep features
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features (the last dimension
    of sim_ftrs)
    """
    batch_size = tf.shape(input=doc_ftrs)[0]
    max_group_size = tf.shape(input=doc_ftrs)[1]

    doc_ftrs = tf.reshape(doc_ftrs, shape=[batch_size, max_group_size, num_deep * num_doc_fields])

    user_ftrs = tf.reshape(user_ftrs, shape=[batch_size, 1, -1]) + tf.zeros([
        batch_size, max_group_size, num_deep * num_user_fields])

    sim_ftrs = tf.concat([doc_ftrs, user_ftrs], axis=2)
    return sim_ftrs


def compute_inner_ftrs_for_user_doc(doc_ftrs, user_ftrs, num_doc_fields, num_user_fields, num_deep):
    """Computes cosine similarity between user and doc fields.

    :param doc_ftrs Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param user_ftrs Shape=[batch_size, num_user_fields, num_deep]
    :param num_doc_fields Number of document fields
    :param num_user_fields Number of user fields
    :param num_deep Number of deep features
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features, which is the last
        dimension of sim_ftrs
    """

    def compute_inner_sim(user_field_ftrs, doc_ftrs):
        """Computes cosine similarity score between user_field_ftrs and doc_ftrs

        :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
        :param user_field_ftrs Shape=[batch_size, num_deep]
        """
        user_field_ftrs = tf.expand_dims(tf.expand_dims(user_field_ftrs, axis=1), axis=1)
        sim_ftrs = tf.reduce_sum(input_tensor=user_field_ftrs * doc_ftrs, axis=-1)
        return sim_ftrs  # [batch_size, group_size, num_doc_fields]

    batch_size = tf.shape(input=doc_ftrs)[0]
    max_group_size = tf.shape(input=doc_ftrs)[1]

    doc_ftrs = tf.nn.l2_normalize(doc_ftrs, axis=-1)
    user_ftrs = tf.nn.l2_normalize(user_ftrs, axis=-1)
    # Shape=[num_user_fields, batch_size, group_size, num_doc_fields]
    sim_ftrs = tf.map_fn(partial(compute_inner_sim, doc_ftrs=doc_ftrs), tf.transpose(a=user_ftrs, perm=[1, 0, 2]))
    # Shape=[batch_size, group_size, num_doc_fields, num_user_fields]
    sim_ftrs = tf.transpose(a=sim_ftrs, perm=[1, 2, 3, 0])
    num_sim_ftrs = NAME_TO_NUM_SIM['inner'](num_doc_fields, num_user_fields)
    # Shape=[batch_size, group_size, num_doc_fields * num_user_fields]
    sim_ftrs = tf.reshape(sim_ftrs, [batch_size, max_group_size, num_sim_ftrs])
    return sim_ftrs


NAME_TO_SIM_FUNC = {'inner': compute_inner_ftrs_for_user_doc,
                    'concat': compute_concat_ftrs_for_user_doc,
                    'hadamard': compute_hadamard_prod_for_user_doc,
                    'diff': compute_elem_diff_for_user_doc}


def compute_sim_ftrs_for_user_doc(doc_ftrs, user_ftrs, num_doc_fields, num_user_fields, sim_func_names, num_deep):
    """Computes similarity between user_ftrs and doc_ftrs with all function names specified in sim_func_names and
        concatenates the results

    :param doc_ftrs Shape=[batch_size, group_size, num_doc_fields, ftr_size]
    :param user_ftrs Shape=[batch_size, num_user_fields, ftr_size]
    :param num_doc_fields Number of document fields
    :param num_user_fields Number of user fields
    :param num_deep Number of deep features
    :param sim_func_names Similarity function names
    """
    sim_ftrs_list = list()

    for name in sim_func_names:
        sim_ftrs = NAME_TO_SIM_FUNC[name](doc_ftrs, user_ftrs, num_doc_fields, num_user_fields, num_deep)
        sim_ftrs_list.append(sim_ftrs)

    return tf.concat(sim_ftrs_list, axis=-1)


class EmbeddingInteractionLayer(tf.keras.layers.Layer):
    """Embedding interaction layer for computing similarity between query, user, and document representations"""

    def __init__(self,
                 num_user_fields_for_interaction,
                 num_doc_fields_for_interaction,
                 has_query,
                 emb_sim_func,
                 deep_ftrs_size):
        super(EmbeddingInteractionLayer, self).__init__()

        self._emb_sim_func = emb_sim_func
        self._deep_ftrs_size = deep_ftrs_size

        self._has_query = has_query
        self._num_user_fields_for_interaction = num_user_fields_for_interaction
        self._num_doc_fields_for_interaction = num_doc_fields_for_interaction

        self.num_sim_ftrs = self._compute_num_sim_ftrs(ftr_size=self._deep_ftrs_size,
                                                       has_query=self._has_query,
                                                       num_user_fields=self._num_user_fields_for_interaction,
                                                       num_doc_fields=self._num_doc_fields_for_interaction,
                                                       emb_sim_func=self._emb_sim_func)

    @staticmethod
    def get_data_size(doc_ftrs):
        """Infers batch size, max group size"""
        assert doc_ftrs is not None, 'doc_ftrs should not be None'
        data_shape = tf.shape(doc_ftrs)
        return data_shape[0], data_shape[1]

    def call(self, inputs, training=None, **kwargs):
        """Returns interaction features for given input"""
        return self.compute_sim_ftrs(query_ftrs=inputs.get(InternalFtrType.QUERY_FTRS, None),
                                     doc_ftrs=inputs.get(InternalFtrType.DOC_FTRS, None),
                                     user_ftrs=inputs.get(InternalFtrType.USER_FTRS, None))

    @staticmethod
    def _compute_num_sim_ftrs(ftr_size, has_query, num_user_fields, num_doc_fields, emb_sim_func):
        """Computes the number of similarity features"""
        # If query is None, use doc_ftrs only and no interaction is performed
        if not has_query and num_user_fields == 0:
            return ftr_size * num_doc_fields

        # Compute interaction between user text/query and document text
        tot_num_user_fields = num_user_fields

        # Treat query as one user field
        tot_num_user_fields += int(has_query)

        return compute_num_sim_ftrs(emb_sim_func, num_doc_fields, tot_num_user_fields, ftr_size)

    def compute_sim_ftrs(self, query_ftrs, doc_ftrs, user_ftrs):
        """
        Computes the interaction features between user/document features and query features.
        When no query and user fields, then return doc_ftrs directly (as concatenation)

        Supported interaction modes (hparams.emb_sim_func) are:
            1. inner: computes cosine similarity between user fields/query and each document field
                num_sim_ftrs=num_doc_fields
            2. concat: concatenates user fields/query and each document field
                num_sim_ftrs=self.ftr_size * (num_user_fields + num_doc_fields)
            3. hadamard: computes Hadamard product between user fields/query and each document field
                num_sim_ftrs=self.ftr_size * num_doc_fields * num_user_fields

        :param query_ftrs Tensor Query features. Shape=[batch_size, ftr_size]
        :param doc_ftrs Tensor Document features. Shape=[batch_size, group_size, num_doc_fields, ftr_size]
        :param user_ftrs Tensor User features. Shape=[batch_size, num_user_fields, ftr_size]

        :return num_sim_ftrs,sim_ftrs  Shape of sim_ftrs is [batch_size, group_size, num_sim_ftrs]. The value of
            num_sim_ftrs is as shown in the instructions above
        """

        num_doc_fields = self._num_doc_fields_for_interaction
        ftr_size = self._deep_ftrs_size

        batch_size, max_group_size = self.get_data_size(doc_ftrs=doc_ftrs)

        # If query is None, use doc_ftrs only and no interaction is performed
        if query_ftrs is None and user_ftrs is None:
            sim_ftrs = tf.reshape(doc_ftrs, shape=[batch_size, max_group_size, ftr_size * num_doc_fields])
            return sim_ftrs

        # Compute interaction between user text/query and document text
        num_user_fields = 0
        tot_user_ftrs = []
        if user_ftrs is not None:
            num_user_fields += self._num_user_fields_for_interaction
            tot_user_ftrs.append(user_ftrs)

        # Treat query as one user field and append it to user field
        if query_ftrs is not None:
            query_ftrs = tf.expand_dims(query_ftrs, axis=1)  # [batch_size, 1, ftr_size]
            tot_user_ftrs.append(query_ftrs)
            num_user_fields += 1

        user_ftrs = tf.concat(tot_user_ftrs, axis=1)  # [batch_size, num_user_fields, ftr_size]
        sim_ftrs = compute_sim_ftrs_for_user_doc(doc_ftrs, user_ftrs, num_doc_fields, num_user_fields, self._emb_sim_func, self._deep_ftrs_size)
        return sim_ftrs


class InteractionLayer(tf.keras.layers.Layer):
    """Interaction layer that computes interaction between following features:
        1. query representations
        2. document representations
        3. user representations
        4. dense features
        5. sparse features
    """

    def __init__(self, use_deep_ftrs, use_wide_ftrs, task_ids, num_hidden, activations,
                 num_user_fields_for_interaction, num_doc_fields_for_interaction, has_query, emb_sim_func, deep_ftrs_size,
                 **kwargs):
        """ Initializes InteractionLayer

        :param use_deep_ftrs: Whether to use deep features (query/user/doc representations)
        :param use_wide_ftrs: Whether to use wide features
        :param task_ids: IDs in multitask training
        :param num_hidden: Hidden layer dimensions, e.g., [100, 200, 20]
        :param activations: Activations of hidden layers, e.g., [tanh, tanh, linear]
        :param num_user_fields_for_interaction: Number of user fields in the user representation
        :param num_doc_fields_for_interaction: Number of document fields in the document representation
        :param has_query: Whether there's query representation from input
        :param emb_sim_func: Functions for similarity computation of query/user/doc representations, e.g. [inner, hadamard, ...]
        :param deep_ftrs_size: Size of last dimension of query/user/doc representations)
        :param kwargs: Args passed to parent Layer
        """
        super().__init__(**kwargs)
        assert use_wide_ftrs or use_deep_ftrs, "Must use either wide/deep features"

        self._use_deep_ftrs = use_deep_ftrs
        self._use_wide_ftrs = use_wide_ftrs
        self._task_ids = task_ids
        self._num_hidden = num_hidden
        self._activations = activations

        self.embedding_interaction_layer = EmbeddingInteractionLayer(
            num_user_fields_for_interaction=num_user_fields_for_interaction,
            num_doc_fields_for_interaction=num_doc_fields_for_interaction,
            has_query=has_query,
            emb_sim_func=emb_sim_func,
            deep_ftrs_size=deep_ftrs_size)
        self.mlps = self.create_mlps(task_ids, self._num_hidden, self._activations)

    def call(self, inputs, **kwargs):
        """Returns interaction features computed between following features
            1. query representations
            2. document representations
            3. user representations
            4. dense features
            5. sparse features

        Query, document and user representations are first fed into an embedding interaction layer to get the interacted sim_ftrs. Then sim_ftrs, dense
          features and sparse features are fed into the MLP layer to get the final interaction features

        :param inputs Map {
            InternalFtrType.QUERY_FTRS: Tensor(dtype=float32, shape=[batch_size, deep_ftr_size]),
            InternalFtrType.DOC_FTRS: Tensor(dtype=float32, shape=[batch_size, list_size, num_doc_fields, deep_ftr_size]),
            InternalFtrType.USER_FTRS: Tensor(dtype=float32, shape=[batch_size, num_user_fields, deep_ftr_size]),
            InternalFtrType.WIDE_FTRS: Tensor(dtype=float32, shape=[batch_size, list_size, wide_ftr_size]),
        }
        :return Map {interaction_ftrs_0: Tensor(dtype=float32, shape=[batch_size, list_size, num_hidden[-1]]), interaction_ftrs_1: ...}
        """
        query_ftrs = inputs.get(InternalFtrType.QUERY_FTRS, None)
        doc_ftrs = inputs.get(InternalFtrType.DOC_FTRS, None)
        user_ftrs = inputs.get(InternalFtrType.USER_FTRS, None)
        wide_ftrs = inputs.get(InternalFtrType.WIDE_FTRS, None)

        all_ftrs = []
        if self._use_deep_ftrs:
            embedding_interaction_layer_inputs = {
                InternalFtrType.QUERY_FTRS: query_ftrs,
                InternalFtrType.DOC_FTRS: doc_ftrs,
                InternalFtrType.USER_FTRS: user_ftrs
            }
            sim_ftrs = self.embedding_interaction_layer(embedding_interaction_layer_inputs)
            all_ftrs.append(sim_ftrs)

        if self._use_wide_ftrs:
            all_ftrs.append(wide_ftrs)

        all_ftrs = tf.concat(all_ftrs, axis=-1)
        return self.compute_mlp_interactions(all_ftrs)

    @staticmethod
    def get_interaction_ftrs_key(index):
        """Returns the key to retrieve interaction outputs of task with the given index """
        return f"{InternalFtrType.INTERACTION_FTRS}_{index}"

    @staticmethod
    def get_interaction_ftrs(inputs, task_ids):
        return {InteractionLayer.get_interaction_ftrs_key(i): inputs[InteractionLayer.get_interaction_ftrs_key(i)] for i, task in enumerate(task_ids)}

    def compute_mlp_interactions(self, all_ftrs):
        """Returns MLP interaction features

        :return Map {interaction_ftrs_0: Tensor(shape=[batch_size, list_size, num_hidden[-1]]), interaction_ftrs_1: ...}
        """
        if self._task_ids is None:
            return {self.get_interaction_ftrs_key(0): self.mlps[0](all_ftrs)}

        return {self.get_interaction_ftrs_key(i): self.mlps[i](all_ftrs) for i, task_id in enumerate(self._task_ids)}

    @staticmethod
    def create_mlps(task_ids, num_hidden, activations):
        """Returns a list of Multi-Layer Perceptrons for given task_ids """
        mlps = []

        # When task_ids is None, treat it as a special case of multitask learning when there's only one task
        if task_ids is None:
            task_ids = [0]

        # Set up layers for each task
        for task_id in task_ids:
            mlps.append(MultiLayerPerceptron(num_hidden, activations, prefix=f'task_{task_id}_'))

        return mlps
