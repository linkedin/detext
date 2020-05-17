import copy
import numpy as np
import os
import tensorflow as tf

from detext.model import deep_match
from detext.model.bert import modeling
from detext.model.rep_model import compute_inner_ftrs_for_usr_doc, compute_concat_ftrs_for_usr_doc, \
    compute_hadamard_prod_for_usr_doc, compute_elem_diff_for_usr_doc


class TestDeepMatch(tf.test.TestCase):
    """Unit test for deep_match.py."""

    # Shared data input across different functions
    query = [[1, 2, 3, 0],
             [2, 4, 3, 1]]

    # Id features
    usr_id_field1 = [[1, 2, 3, 0],
                     [2, 4, 3, 1]]
    usr_id_field2 = [[1, 2, 3, 0],
                     [2, 4, 3, 1]]
    doc_id_field1 = [[[1, 2, 3, 0],
                      [2, 4, 3, 1],
                      [2, 4, 3, 1]],
                     [[2, 4, 3, 1],
                      [1, 3, 3, 1],
                      [5, 6, 1, 0]]]
    doc_id_field2 = [[[1, 2, 3, 0],
                      [20, 5, 3, 1],
                      [2, 4, 3, 1]],
                     [[20, 5, 3, 1],
                      [1, 3, 3, 1],
                      [5, 6, 0, 0]]]

    # Text fields
    doc_field1 = [[[2, 4, 3, 1],
                   [2, 4, 3, 1],
                   [0, 0, 0, 0]],
                  [[2, 4, 3, 1],
                   [1, 3, 3, 1],
                   [5, 6, 1, 0]]]
    doc_field2 = [[[1, 2, 3, 0],
                   [20, 5, 3, 1],
                   [0, 0, 0, 0]],
                  [[20, 5, 3, 1],
                   [1, 3, 3, 0],
                   [5, 6, 0, 0]]]

    # Wide features, group_size and labels
    wide_ftrs = np.random.rand(2, 3, 5)
    group_size = [2, 3]
    labels = [[1, 0, 1], [0, 0, 1]]

    # Config and hparams
    res_dir = os.path.dirname(__file__) + '/../resources'
    bert_config_file = res_dir + '/bert_config.json'
    hparams = tf.contrib.training.HParams(
        we_trainable=True,
        use_tfr_loss=False,
        bert_checkpoint=None,
        bert_config_file=bert_config_file,
        bert_config=modeling.BertConfig.from_json_file(bert_config_file),
        emb_sim_func=['inner'],
        elem_rescale=False,
        filter_window_sizes=[1, 2, 3],
        ftr_ext='bert',
        ltr_loss_fn='pairwise',
        num_units=50,
        num_doc_fields=2,
        num_filters=100,
        num_hidden=[10],
        num_wide=5,
        num_usr_fields=0,
        pad_id=0,
        use_wide=True,
        use_deep=True,
        vocab_size=30000,
        we_file=None,
        lambda_metric=None,
        ndcg_topk=10,
        explicit_empty=True,
        num_classes=1,
        # Id features
        num_doc_id_fields=0,
        num_usr_id_fields=0,
        num_units_for_id_ftr=10,
        vocab_size_for_id_ftr=30,
        we_trainable_for_id_ftr=True,
        we_file_for_id_ftr=None,
        pad_id_for_id_ftr=0,
        use_doc_projection=False,
        use_usr_projection=False,
        task_ids=None,
    )

    def _get_constant_usr_fields(self):
        """Returns constant tensor to be used as user text fields"""
        return [tf.constant(self.query, dtype=tf.int32), tf.constant(self.query, dtype=tf.int32)]

    def _get_constant_doc_fields(self):
        """Returns constant tensor to be used as doc text fields"""
        doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
        doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
        doc_fields = [doc_field1, doc_field2]
        return doc_fields

    def _get_constant_usr_id_fields(self):
        """Returns constant tensor to be used as usr id fields"""
        usr_id_field1 = tf.constant(self.usr_id_field1, dtype=tf.int32)
        usr_id_field2 = tf.constant(self.usr_id_field2, dtype=tf.int32)
        usr_id_fields = [usr_id_field1, usr_id_field2]
        return usr_id_fields

    def _get_constant_doc_id_fields(self):
        """Returns constant tensor to be used as doc id fields"""
        doc_id_field1 = tf.constant(self.doc_id_field1, dtype=tf.int32)
        doc_id_field2 = tf.constant(self.doc_id_field2, dtype=tf.int32)
        doc_id_fields = [doc_id_field1, doc_id_field2]
        return doc_id_fields

    def testDeepMatch(self):
        """Tests DeepMatch outputs"""
        hparams = self.hparams
        query = tf.constant(self.query, dtype=tf.int32)
        doc_fields = self._get_constant_doc_fields()

        wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)

        dm = deep_match.DeepMatch(query,
                                  wide_ftrs,
                                  doc_fields,
                                  hparams,
                                  tf.estimator.ModeKeys.EVAL)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            scores = sess.run(dm.scores)

            # Check sizes and shapes
            self.assertAllEqual(scores.shape, [2, 3])

    def testDeepMatchNoQuery(self):
        """Tests DeepMatch with query as None.

        Two conditions must be met when query is None:
        1. query_ftrs of model.text_encoding_model must be None
        2. final_ftrs should only contain information about doc_ftrs and wide_ftrs
        """
        hparams = copy.copy(self.hparams)
        # ftr_ext = cnn
        with tf.Graph().as_default():
            query = None
            doc_fields = self._get_constant_doc_fields()
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
            setattr(hparams, 'ftr_ext', 'cnn')
            # Test no query field when ftr_ext = bert
            dm = deep_match.DeepMatch(query,
                                      wide_ftrs,
                                      doc_fields,
                                      hparams,
                                      tf.estimator.ModeKeys.EVAL)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                wide_ftrs = wide_ftrs.eval()

                # final_ftrs should only contain wide_ftrs and doc_fields
                doc_ftrs_size_after_cnn = len(doc_fields) * len(hparams.filter_window_sizes) * hparams.num_filters
                self.assertAllEqual(dm.final_ftrs.eval().shape[-1],
                                    doc_ftrs_size_after_cnn + wide_ftrs.shape[-1] + len(doc_fields))

                # query_ftrs of model.text_encoding_model must be None
                self.assertAllEqual(dm.deep_ftr_model.text_encoding_model.query_ftrs, None)

    def testDeepMatchNaNFtrs(self):
        """Tests DeepMatch outputs"""
        query = tf.constant(self.query, dtype=tf.int32)
        doc_fields = self._get_constant_doc_fields()

        nan_wide_ftrs = np.copy(self.wide_ftrs)
        nan_removed_wide_ftrs = np.copy(self.wide_ftrs)

        nan_wide_ftrs[0] = np.nan
        nan_removed_wide_ftrs[0] = 0

        wide_ftrs = tf.constant(nan_wide_ftrs, dtype=tf.float32)
        hparams = self.hparams

        dm = deep_match.DeepMatch(query,
                                  wide_ftrs,
                                  doc_fields,
                                  hparams,
                                  tf.estimator.ModeKeys.EVAL)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            wide_ftrs_dm, scores = sess.run(
                [dm._wide_ftrs,
                 dm.scores]
            )

            # check sizes and shapes
            self.assertAllClose(wide_ftrs_dm, nan_removed_wide_ftrs)
            self.assertAllEqual(scores.shape, [2, 3])

    def testDeepMatchWithUsrField(self):
        """Tests DeepMatch with user fields."""

        hparams = copy.copy(self.hparams)
        # ftr_ext = bert
        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
            doc_fields = self._get_constant_doc_fields()
            usr_fields = self._get_constant_usr_fields()
            setattr(hparams, 'emb_sim_func', ['concat'])
            setattr(hparams, 'num_usr_fields', len(usr_fields))

            # Test no query field when ftr_ext = bert
            dm = deep_match.DeepMatch(query,
                                      wide_ftrs,
                                      doc_fields,
                                      hparams,
                                      tf.estimator.ModeKeys.EVAL,
                                      usr_fields=usr_fields)

            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                wide_ftrs = wide_ftrs.eval()

                # final_ftrs should contain information about wide_ftrs user_fields and doc_fields
                self.assertAllEqual(dm.final_ftrs.eval().shape[-1], (len(doc_fields) + len(usr_fields) + 1) * (
                    dm.deep_ftr_model.ftr_size + int(hparams.explicit_empty)) + wide_ftrs.shape[-1])

    def testDeepMatchWithIdField(self):
        """Tests DeepMatch with id fields"""
        hparams = copy.copy(self.hparams)
        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)

            doc_fields = self._get_constant_doc_fields()
            usr_fields = self._get_constant_usr_fields()

            doc_id_fields = self._get_constant_doc_id_fields()
            usr_id_fields = self._get_constant_usr_id_fields()

            setattr(hparams, 'emb_sim_func', ['concat', 'inner'])
            setattr(hparams, 'num_usr_fields', len(usr_fields))

            setattr(hparams, 'num_doc_id_fields', len(doc_id_fields))
            setattr(hparams, 'num_usr_id_fields', len(usr_id_fields))

            # Test no query field when ftr_ext = bert
            dm = deep_match.DeepMatch(query,
                                      wide_ftrs,
                                      doc_fields,
                                      hparams,
                                      tf.estimator.ModeKeys.EVAL,
                                      usr_fields=usr_fields,
                                      usr_id_fields=usr_id_fields,
                                      doc_id_fields=doc_id_fields)

            with self.test_session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                wide_ftrs = wide_ftrs.eval()

                ftr_size = dm.deep_ftr_model.ftr_size
                concat_ftr_size = (len(doc_fields) + len(usr_fields) + 1) * (ftr_size + int(hparams.explicit_empty)) + \
                                  (len(usr_id_fields) + len(doc_id_fields)) * ftr_size
                inner_ftr_size = (len(usr_fields) + len(usr_id_fields) + 1) * (len(doc_fields) + len(doc_id_fields))
                # final_ftrs should contain information about wide_ftrs user_fields and doc_fields
                self.assertAllEqual(dm.final_ftrs.eval().shape[-1],
                                    concat_ftr_size + inner_ftr_size + wide_ftrs.shape[-1])

    def testComputeSim(self):
        """Tests correctness of similarity computation functions"""
        with tf.Graph().as_default():
            num_doc_fields = 2
            num_usr_fields = 1
            num_deep = 5

            # Input
            usr_ftrs = tf.constant([[[1, 2, 3, 4, 5]]], dtype=tf.float32)
            doc_ftrs = tf.constant([[[[1, 2, 3, 4, 5],
                                      [-5, -4, 3, 2, 1]]]], dtype=tf.float32)

            # Similarity features
            inner_sim_ftrs, _ = compute_inner_ftrs_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields,
                                                               num_deep)
            concat_sim_ftrs, _ = compute_concat_ftrs_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields,
                                                                 num_deep)
            hadamard_sim_ftrs, _ = compute_hadamard_prod_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields,
                                                                     num_deep)
            diff_sim_ftrs, _ = compute_elem_diff_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields,
                                                             num_deep)
            with tf.Session() as sess:
                inner_sim_ftrs_v, concat_sim_ftrs_v, hadamard_sim_ftrs_v, diff_sim_ftrs_v = sess.run(
                    [inner_sim_ftrs, concat_sim_ftrs, hadamard_sim_ftrs, diff_sim_ftrs])
                self.assertAllClose(inner_sim_ftrs_v, [[[1., 0.16363636363636364]]])
                self.assertAllEqual(concat_sim_ftrs_v, [[[1, 2, 3, 4, 5,
                                                          -5, -4, 3, 2, 1,
                                                          1, 2, 3, 4, 5]]])
                self.assertAllEqual(hadamard_sim_ftrs_v, [[[1, 4, 9, 16, 25,
                                                            -5, -8, 9, 8, 5]]])
                self.assertAllEqual(diff_sim_ftrs, [[[0, 0, 0, 0, 0,
                                                      6, 6, 0, 2, 4]]])

    def testDeepMatchClassification(self):
        """Tests DeepMatch for classification outputs"""
        query = tf.constant(self.query, dtype=tf.int32)
        doc_field1 = tf.constant(np.random.rand(2, 1, 4), dtype=tf.int32)
        doc_field2 = tf.constant(np.random.rand(2, 1, 4), dtype=tf.int32)
        doc_fields = [doc_field1, doc_field2]
        wide_ftrs = np.random.rand(2, 1, 5)
        wide_ftrs = tf.constant(wide_ftrs, dtype=tf.float32)
        hparamscp = copy.copy(self.hparams)
        hparamscp.num_classes = 7
        dm = deep_match.DeepMatch(query,
                                  wide_ftrs,
                                  doc_fields,
                                  hparamscp,
                                  tf.estimator.ModeKeys.EVAL)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            scores = sess.run(
                dm.scores
            )
        # Check sizes and shapes
        self.assertAllEqual(scores.shape, [2, hparamscp.num_classes])

    def testDeepMatchClassificationNoQueryNoWide(self):
        """Tests DeepMatch for classification outputs"""
        doc_field1 = tf.constant(np.random.rand(2, 1, 4), dtype=tf.int32)
        doc_field2 = tf.constant(np.random.rand(2, 1, 4), dtype=tf.int32)
        doc_fields = [doc_field1, doc_field2]
        hparamscp = copy.copy(self.hparams)
        hparamscp.num_classes = 7
        dm = deep_match.DeepMatch(None,
                                  None,
                                  doc_fields,
                                  hparamscp,
                                  tf.estimator.ModeKeys.EVAL)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            scores = sess.run(
                dm.scores
            )
        # Check sizes and shapes
        self.assertAllEqual(scores.shape, [2, hparamscp.num_classes])

    def testDeepMatcDocUsrProjection(self):
        """Tests DeepMatch with doc/usr fields projection."""
        hparams = copy.copy(self.hparams)
        # ftr_ext = cnn
        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            doc_fields = self._get_constant_doc_fields()
            usr_fields = self._get_constant_usr_fields()
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
            setattr(hparams, 'use_doc_projection', True)
            setattr(hparams, 'use_usr_projection', True)
            setattr(hparams, 'num_usr_fields', len(usr_fields))
            setattr(hparams, 'explicit_empty', False)
            # num_sim_ftrs should be doc projection size* (query + user projection size)
            expected_num_sim_ftrs = 1 * (1 + 1)
            # Test no query field when ftr_ext = bert
            dm = deep_match.DeepMatch(query,
                                      wide_ftrs,
                                      doc_fields,
                                      hparams,
                                      tf.estimator.ModeKeys.EVAL,
                                      usr_fields=usr_fields)

            self.assertAllEqual(dm.deep_ftr_model.num_sim_ftrs, expected_num_sim_ftrs)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self.assertAllEqual(dm.deep_ftr_model.sim_ftrs.eval().shape, self.group_size + [expected_num_sim_ftrs])

    def testDeepMatchMultitaskRanking(self):
        """Tests DeepMatch with multitask ranking"""
        query = tf.constant(self.query, dtype=tf.int32)
        doc_fields = self._get_constant_doc_fields()
        wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
        task_id_field = tf.constant([1, 0])

        hparamscp = copy.copy(self.hparams)
        hparamscp.task_ids = {'0': 0.2, '1': 0.8}

        dm = deep_match.DeepMatch(query=query,
                                  wide_ftrs=wide_ftrs,
                                  doc_fields=doc_fields,
                                  hparams=hparamscp,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  task_id_field=task_id_field)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            scores = sess.run(dm.scores)

            # Check sizes and shapes
            self.assertAllEqual(scores.shape, [2, 3])


if __name__ == "__main__":
    tf.test.main()
