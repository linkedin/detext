import tensorflow as tf

from detext.model import lstm_lm_model


class TestLstmLmModel(tf.test.TestCase):
    """Unit test for deep_model.py."""
    query = [[1, 2, 3, 0],
             [2, 4, 3, 1]]
    doc_field1 = [[[1, 2, 3, 0],
                   [2, 4, 3, 1],
                   [2, 4, 3, 1]],
                  [[2, 4, 3, 1],
                   [1, 3, 3, 1],
                   [5, 6, 1, 0]]]
    doc_field2 = [[[1, 2, 3, 0],
                   [20, 5, 3, 1],
                   [2, 4, 3, 1]],
                  [[20, 5, 3, 1],
                   [1, 3, 3, 1],
                   [5, 6, 0, 0]]]

    def testLstmLmModel(self):
        """test LstmLmFtrExt outputs"""

        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            usr_fields = [query, query, query]

            hparams = tf.contrib.training.HParams(
                we_trainable=True,
                use_tfr_loss=False,
                num_units=100,
                num_doc_fields=2,
                num_wide=5,
                vocab_size=30000,
                we_file=None,
                bert_checkpoint=None,
                use_wide=False,
                use_deep=True,
                num_hidden=[0],
                pad_id=0,
                unit_type='lstm',
                num_layers=1,
                num_residual_layers=0,
                forget_bias=0.5,
                random_seed=11,
                rnn_dropout=0.,
                bidirectional=True,
                normalized_lm=False,
            )

            model = lstm_lm_model.LstmLmModel(query,
                                              doc_fields,
                                              usr_fields,
                                              hparams=hparams,
                                              mode=tf.estimator.ModeKeys.EVAL)
            text_ftr_size = 1

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                query_ftrs, doc_ftrs, usr_ftrs = sess.run((model.query_ftrs, model.doc_ftrs, model.usr_ftrs))

                self.assertEquals(text_ftr_size, model.text_ftr_size)
                self.assertAllEqual(query_ftrs.shape, [2, text_ftr_size])
                self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, text_ftr_size])
                self.assertAllEqual(usr_ftrs.shape, [2, 3, text_ftr_size])

                # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field
                self.assertAllEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
                # 1st query, 1st doc, 1st field should NOT be the same as 1st query, 2nd doc, 1st field
                self.assertNotAllClose(doc_ftrs[0, 0, 0], doc_ftrs[0, 1, 0])

    def testLstmLmModelNormalized(self):
        """Test LstmLmFtrExt outputs for normalized language model"""

        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            usr_fields = [query, query, query]

            hparams = tf.contrib.training.HParams(
                we_trainable=True,
                use_tfr_loss=False,
                num_units=100,
                num_doc_fields=2,
                num_wide=5,
                vocab_size=30000,
                we_file=None,
                bert_checkpoint=None,
                use_wide=False,
                use_deep=True,
                normalized_lm=True,
                num_hidden=[0],
                pad_id=0,
                unit_type='lstm',
                num_layers=1,
                num_residual_layers=0,
                forget_bias=0.5,
                random_seed=11,
                rnn_dropout=0.,
                bidirectional=True,
            )

            model = lstm_lm_model.LstmLmModel(query,
                                              doc_fields,
                                              usr_fields,
                                              hparams=hparams,
                                              mode=tf.estimator.ModeKeys.EVAL)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                query_ftrs, doc_ftrs, usr_ftrs = sess.run((model.query_ftrs, model.doc_ftrs, model.usr_ftrs))

                self.assertAllEqual(query_ftrs.shape, [2, 1])
                self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, 1])
                self.assertAllEqual(usr_ftrs.shape, [2, 3, 1])

                # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field
                self.assertAlmostEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
                # 1st query, 1st doc, 1st field should NOT be the same as 1st query, 2nd doc, 1st field
                self.assertNotEquals(doc_ftrs[0, 0, 0], doc_ftrs[0, 1, 0])


if __name__ == "__main__":
    tf.test.main()
