import numpy as np
import os
import tensorflow as tf

from detext.model import deep_match
from detext.model.bert import modeling
from detext.model.rep_model import compute_inner_ftrs_for_usr_doc, compute_concat_ftrs_for_usr_doc, \
    compute_hadamard_prod_for_usr_doc


class TestDeepMatch(tf.test.TestCase):
    """Unit test for deep_match.py."""

    # Shared data input across different functions
    query = [[1, 2, 3, 0],
             [2, 4, 3, 1]]

    # Id features
    usr_id_ftr1 = [1, 5]
    usr_id_size1 = 6
    usr_id_ftr2 = [312, 234]
    usr_id_size2 = 350

    doc_id_ftr1 = [[1, 3, 2],
                   [1, 3, 2]]
    doc_id_size1 = 5
    doc_id_ftr2 = [[100, 30, 20],
                   [100, 32, 32]]
    doc_id_size2 = 200

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
        explicit_empty=True
    )

    def testDeepMatch(self):
        """Tests DeepMatch outputs"""
        query = tf.constant(self.query, dtype=tf.int32)
        doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
        doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
        doc_fields = [doc_field1, doc_field2]

        wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
        hparams = self.hparams

        dm = deep_match.DeepMatch(query,
                                  wide_ftrs,
                                  doc_fields,
                                  hparams,
                                  tf.estimator.ModeKeys.EVAL)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            scores = sess.run(
                dm.scores
            )

            # Check sizes and shapes
            self.assertAllEqual(scores.shape, [2, 3])

    def testDeepMatchNoQuery(self):
        """Tests DeepMatch with query as None.

        Two conditions must be met when query is None:
        1. query_ftrs of model.scoring_model must be None
        2. final_ftrs should only contain information about doc_ftrs and wide_ftrs
        """
        # ftr_ext = bert
        with tf.Graph().as_default():
            query = None
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
            hparams = self.hparams

            # Test no query field when ftr_ext = bert
            dm = deep_match.DeepMatch(query,
                                      wide_ftrs,
                                      doc_fields,
                                      hparams,
                                      tf.estimator.ModeKeys.EVAL)

            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                wide_ftrs = wide_ftrs.eval()

                # final_ftrs should only contain wide_ftrs and doc_fields
                self.assertAllEqual(dm.final_ftrs.eval().shape[-1], len(doc_fields) * (50 + 1) + wide_ftrs.shape[-1])

                # query_ftrs of model.scoring_model must be None
                self.assertAllEqual(dm.deep_ftr_model.scoring_model.query_ftrs, None)

        # ftr_ext = cnn
        with tf.Graph().as_default():
            query = None
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
            hparams = self.hparams
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

                # query_ftrs of model.scoring_model must be None
                self.assertAllEqual(dm.deep_ftr_model.scoring_model.query_ftrs, None)
            setattr(hparams, 'ftr_ext', 'bert')

    def testDeepMatchNaNFtrs(self):
        """Tests DeepMatch outputs"""
        query = tf.constant(self.query, dtype=tf.int32)
        doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
        doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
        doc_fields = [doc_field1, doc_field2]

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
        """Tests DeepMatch with user fields.

        Two conditions must be met when query is None:
        """
        # ftr_ext = bert
        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
            usr_fields = [query, query]

            hparams = self.hparams
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
                self.assertAllEqual(dm.final_ftrs.eval().shape[-1],
                                    (len(doc_fields) + len(usr_fields) + 1) * (50 + 1) + wide_ftrs.shape[-1])

        # ftr_ext = cnn
        with tf.Graph().as_default():
            query = None
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            wide_ftrs = tf.constant(self.wide_ftrs, dtype=tf.float32)
            usr_fields = [tf.constant(self.query, dtype=tf.int32), tf.constant(self.query, dtype=tf.int32)]

            hparams = self.hparams
            setattr(hparams, 'ftr_ext', 'cnn')
            setattr(hparams, 'num_usr_fields', len(usr_fields))
            # Test no query field when ftr_ext = bert
            dm = deep_match.DeepMatch(query,
                                      wide_ftrs,
                                      doc_fields,
                                      hparams,
                                      tf.estimator.ModeKeys.EVAL,
                                      usr_fields=usr_fields)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                wide_ftrs = wide_ftrs.eval()

                # final_ftrs should only contain wide_ftrs and doc_fields
                doc_ftrs_size_after_cnn = len(doc_fields) * len(hparams.filter_window_sizes) * hparams.num_filters
                usr_ftrs_size_after_cnn = len(usr_fields) * len(hparams.filter_window_sizes) * hparams.num_filters
                self.assertAllEqual(dm.final_ftrs.eval().shape[-1],
                                    doc_ftrs_size_after_cnn + usr_ftrs_size_after_cnn + wide_ftrs.shape[-1] + len(doc_fields) + len(usr_fields))

                # query_ftrs of model.scoring_model must be None
                self.assertAllEqual(dm.deep_ftr_model.scoring_model.query_ftrs, None)
            setattr(hparams, 'ftr_ext', 'bert')
            setattr(hparams, 'emb_sim_func', ['inner'])
            setattr(hparams, 'num_usr_fields', 0)

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

            with tf.Session() as sess:
                inner_sim_ftrs_v, concat_sim_ftrs_v, hadamard_sim_ftrs_v = sess.run(
                    [inner_sim_ftrs, concat_sim_ftrs, hadamard_sim_ftrs])
                self.assertAllClose(inner_sim_ftrs_v, [[[1., 0.16363636363636364]]])
                self.assertAllEqual(concat_sim_ftrs_v, [[[1, 2, 3, 4, 5,
                                                          -5, -4, 3, 2, 1,
                                                          1, 2, 3, 4, 5]]])
                self.assertAllEqual(hadamard_sim_ftrs_v, [[[1, 4, 9, 16, 25,
                                                            -5, -8, 9, 8, 5]]])


if __name__ == "__main__":
    tf.test.main()
