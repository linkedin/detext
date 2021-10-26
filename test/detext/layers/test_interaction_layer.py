import copy
from itertools import product

import tensorflow as tf

from detext.layers import interaction_layer
from detext.utils.parsing_utils import InternalFtrType
from detext.utils.testing.data_setup import DataSetup


class TestEmbeddingInteractionLayer(tf.test.TestCase, DataSetup):
    """Unit test for interaction_layer.py"""
    deep_ftr_size = 3
    batch_size = 2
    max_group_size = 4
    num_doc_fields = 3
    num_user_fields = 2
    atol = 1e-4

    has_query = True
    emb_sim_func = ['concat']

    query_ftrs = tf.random.uniform([batch_size, deep_ftr_size])
    user_ftrs = tf.random.uniform([batch_size, num_user_fields, deep_ftr_size])
    doc_ftrs = tf.random.uniform([batch_size, max_group_size, num_doc_fields, deep_ftr_size])
    dense_ftrs = tf.random.uniform([batch_size, max_group_size, deep_ftr_size])

    embedding_interaction_layer_input = {
        InternalFtrType.QUERY_FTRS: query_ftrs,
        InternalFtrType.USER_FTRS: user_ftrs,
        InternalFtrType.DOC_FTRS: doc_ftrs
    }

    interaction_layer_input = {
        **embedding_interaction_layer_input,
        InternalFtrType.WIDE_FTRS: dense_ftrs
    }
    num_hidden = [20, 30, 4]
    activations = ['tanh'] * len(num_hidden)

    def testInteractionLayer(self):
        """Tests InteractionLayer"""
        use_deep_ftrs_list = [True, False]
        use_dense_ftrs_list = [True, False]
        task_ids_list = [None, [0], [3, 23, 222]]
        for use_deep_ftrs, use_dense_ftrs, task_ids in product(use_deep_ftrs_list, use_dense_ftrs_list, task_ids_list):
            inputs = copy.copy(self.interaction_layer_input)
            if not use_deep_ftrs:
                for ftr_type in [InternalFtrType.QUERY_FTRS, InternalFtrType.DOC_FTRS, InternalFtrType.USER_FTRS]:
                    inputs.pop(ftr_type)

            if not use_dense_ftrs:
                inputs.pop(InternalFtrType.WIDE_FTRS)

            if use_deep_ftrs or use_dense_ftrs:
                self._testInteractionLayer(inputs, use_deep_ftrs, use_dense_ftrs, task_ids)

    def _testInteractionLayer(self, inputs, use_deep_ftrs, use_dense_ftrs, task_ids):
        """Tests InteractionLayer given settings"""
        layer = interaction_layer.InteractionLayer(
            use_deep_ftrs=use_deep_ftrs,
            use_wide_ftrs=use_dense_ftrs,
            task_ids=task_ids,
            num_hidden=self.num_hidden,
            activations=self.activations,

            num_user_fields_for_interaction=self.num_user_fields,
            num_doc_fields_for_interaction=self.num_doc_fields,
            has_query=self.has_query,
            emb_sim_func=self.emb_sim_func,
            deep_ftrs_size=self.deep_ftr_size)
        outputs = layer(inputs)
        task_ids = [0] if task_ids is None else task_ids
        self.assertAllEqual(len(outputs), len(task_ids))
        for i in range(len(task_ids)):
            self.assertAllEqual(tf.shape(outputs[interaction_layer.InteractionLayer.get_interaction_ftrs_key(i)]),
                                [self.batch_size, self.max_group_size, self.num_hidden[-1]])

    def testEmbeddingInteractionLayer(self):
        """Tests EmbeddingInteractionLayer"""
        layer = interaction_layer.EmbeddingInteractionLayer(
            num_user_fields_for_interaction=self.num_user_fields,
            num_doc_fields_for_interaction=self.num_doc_fields,
            has_query=self.has_query,
            emb_sim_func=self.emb_sim_func,
            deep_ftrs_size=self.deep_ftr_size)

        sim_ftrs = layer(self.embedding_interaction_layer_input)
        self.assertAllEqual(tf.shape(sim_ftrs),
                            [self.batch_size, self.max_group_size, (self.num_doc_fields + self.num_user_fields + self.has_query) * self.deep_ftr_size])

        # Test when there are only doc_ftrs
        layer = interaction_layer.EmbeddingInteractionLayer(
            num_user_fields_for_interaction=0,
            num_doc_fields_for_interaction=self.num_doc_fields,
            has_query=False,
            emb_sim_func=self.emb_sim_func,
            deep_ftrs_size=self.deep_ftr_size)
        inputs = {InternalFtrType.DOC_FTRS: self.embedding_interaction_layer_input[InternalFtrType.DOC_FTRS]}
        sim_ftrs = layer(inputs)
        self.assertAllEqual(tf.shape(sim_ftrs),
                            [self.batch_size, self.max_group_size, self.num_doc_fields * self.deep_ftr_size])

    def testEmbeddingInteractionUtils(self):
        """Tests interaction utils"""
        doc_ftrs_lst = [tf.ones([self.batch_size, self.max_group_size, self.num_doc_fields, self.deep_ftr_size]),
                        tf.zeros([self.batch_size, self.max_group_size, self.num_doc_fields, self.deep_ftr_size]),
                        tf.zeros([self.batch_size, self.max_group_size, self.num_doc_fields, self.deep_ftr_size])]
        user_ftrs_lst = [tf.ones([self.batch_size, self.num_user_fields, self.deep_ftr_size]),
                         tf.ones([self.batch_size, self.num_user_fields, self.deep_ftr_size]),
                         tf.zeros([self.batch_size, self.num_user_fields, self.deep_ftr_size])]

        # Unit test for hadamard similarity
        sim_func_names = ['hadamard']
        expected_sim_ftrs_lst = [
            tf.ones([self.batch_size, self.max_group_size, self.num_doc_fields * self.num_user_fields * self.deep_ftr_size]),
            tf.zeros([self.batch_size, self.max_group_size, self.num_doc_fields * self.num_user_fields * self.deep_ftr_size]),
            tf.zeros([self.batch_size, self.max_group_size, self.num_doc_fields * self.num_user_fields * self.deep_ftr_size])]
        expected_num_sim_ftrs = self.num_doc_fields * self.num_user_fields * self.deep_ftr_size

        assert len(doc_ftrs_lst) == len(user_ftrs_lst) == len(expected_sim_ftrs_lst), 'There must be same test cases for every test params'
        for doc_ftrs, user_ftrs, expected_sim_ftrs in zip(doc_ftrs_lst, user_ftrs_lst, expected_sim_ftrs_lst):
            self._testEmbeddingInteractionUtils(doc_ftrs, user_ftrs, sim_func_names, expected_sim_ftrs, expected_num_sim_ftrs)

        # Unit test for cosine interaction similarity
        sim_func_names = ['inner']
        expected_sim_ftrs_lst_inner = [
            tf.ones([self.batch_size, self.max_group_size, self.num_doc_fields * self.num_user_fields]),
            tf.zeros([self.batch_size, self.max_group_size, self.num_doc_fields * self.num_user_fields]),
            tf.zeros([self.batch_size, self.max_group_size, self.num_doc_fields * self.num_user_fields])]
        expected_num_sim_ftrs_inner = self.num_doc_fields * self.num_user_fields
        assert len(doc_ftrs_lst) == len(user_ftrs_lst) == len(expected_sim_ftrs_lst_inner), 'There must be same test cases for every test params'
        for doc_ftrs, user_ftrs, expected_sim_ftrs in zip(doc_ftrs_lst, user_ftrs_lst, expected_sim_ftrs_lst_inner):
            self._testEmbeddingInteractionUtils(doc_ftrs, user_ftrs, sim_func_names, expected_sim_ftrs, expected_num_sim_ftrs_inner)

        # Unit test for concatenation
        sim_func_names = ['concat']
        expected_sim_ftrs_lst_concat = [
            tf.ones([self.batch_size, self.max_group_size, (self.num_doc_fields + self.num_user_fields) * self.deep_ftr_size]),
            tf.concat([tf.zeros([self.batch_size, self.max_group_size, self.num_doc_fields * self.deep_ftr_size]),
                       tf.ones([self.batch_size, self.max_group_size, self.num_user_fields * self.deep_ftr_size])], axis=-1),
            tf.zeros([self.batch_size, self.max_group_size, (self.num_doc_fields + self.num_user_fields) * self.deep_ftr_size])]
        expected_num_sim_ftrs_concat = (self.num_doc_fields + self.num_user_fields) * self.deep_ftr_size
        assert len(doc_ftrs_lst) == len(user_ftrs_lst) == len(expected_sim_ftrs_lst_concat), 'There must be same test cases for every test params'
        for doc_ftrs, user_ftrs, expected_sim_ftrs in zip(doc_ftrs_lst, user_ftrs_lst, expected_sim_ftrs_lst_concat):
            self._testEmbeddingInteractionUtils(doc_ftrs, user_ftrs, sim_func_names, expected_sim_ftrs, expected_num_sim_ftrs_concat)

        # Unit test for multiple interaction
        sim_func_names = ['inner', 'concat']
        expected_sim_ftrs_lst = [tf.concat([inner_sim_ftrs, concat_sim_ftrs], axis=-1) for inner_sim_ftrs, concat_sim_ftrs in
                                 zip(expected_sim_ftrs_lst_inner, expected_sim_ftrs_lst_concat)]
        expected_num_sim_ftrs = expected_num_sim_ftrs_inner + expected_num_sim_ftrs_concat
        assert len(doc_ftrs_lst) == len(user_ftrs_lst) == len(expected_sim_ftrs_lst), 'There must be same test cases for every test params'
        for doc_ftrs, user_ftrs, expected_sim_ftrs in zip(doc_ftrs_lst, user_ftrs_lst, expected_sim_ftrs_lst):
            self._testEmbeddingInteractionUtils(doc_ftrs, user_ftrs, sim_func_names, expected_sim_ftrs, expected_num_sim_ftrs)

    def _testEmbeddingInteractionUtils(self, doc_ftrs, user_ftrs, sim_func_names, expected_sim_ftrs, expected_num_sim_ftrs):
        """Tests interaction layer given input """
        sim_ftrs = interaction_layer.compute_sim_ftrs_for_user_doc(doc_ftrs, user_ftrs, self.num_doc_fields, self.num_user_fields, sim_func_names,
                                                                   self.deep_ftr_size)
        num_sim_ftrs = interaction_layer.compute_num_sim_ftrs(sim_func_names, self.num_doc_fields, self.num_user_fields, self.deep_ftr_size)

        self.assertAllClose(sim_ftrs, expected_sim_ftrs, atol=self.atol)
        self.assertEqual(num_sim_ftrs, expected_num_sim_ftrs)


if __name__ == '__main__':
    tf.test.main()
