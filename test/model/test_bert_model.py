import os
import tensorflow as tf

from detext.model import bert_model
from detext.model.bert import modeling


class TestBERTFtrExt(tf.test.TestCase):
    """Unit test for bert_model.py."""
    query = [[1, 2, 3],
             [2, 4, 3]]
    doc_field1 = [[[1, 2, 3, 0],
                   [2, 4, 3, 1],
                   [0, 0, 0, 0]],
                  [[2, 4, 3, 1],
                   [1, 3, 3, 1],
                   [1, 3, 3, 1]]]
    doc_field2 = [[[1, 2, 3, 0],
                   [2, 4, 3, 1],
                   [0, 0, 0, 0]],
                  [[20, 5, 3, 1],
                   [5, 6, 1, 1],
                   [5, 6, 1, 1]]]

    res_dir = os.path.dirname(__file__) + '/../resources'
    bert_config_file = res_dir + '/bert_config.json'
    hparams = tf.contrib.training.HParams(
        we_trainable=True,
        use_tfr_loss=False,
        bert_config_file=bert_config_file,
        bert_config=modeling.BertConfig.from_json_file(bert_config_file),
        filter_window_sizes=[1, 2, 3],
        num_units=100,
        num_filters=100,
        num_doc_fields=2,
        num_wide=5,
        num_hidden=0,
        pad_id=0,
        vocab_size=30000,
        emb_sim_func='outer',
        use_wide=False,
    )

    def testBERTFtrExt(self):
        """test BERTFtrExt outputs and intermediate results"""

        query = tf.constant(self.query, dtype=tf.int32)
        doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
        doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
        doc_fields = [doc_field1, doc_field2]
        usr_fields = None

        hparams = self.hparams

        bfe = bert_model.BertModel(
            query,
            doc_fields,
            usr_fields,
            hparams=hparams,
            mode=tf.estimator.ModeKeys.EVAL)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            query_ftrs, doc_ftrs, bert_input_ids = \
                sess.run((bfe.query_ftrs, bfe.doc_ftrs, bfe.bert_input_ids))
            # check query/doc ftr shape
            self.assertAllEqual(query_ftrs.shape, [2, 100])
            self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, bfe.text_ftr_size])

            self.assertAllEqual(bfe.usr_ftrs, None)
            # check bert input ids
            self.assertAllEqual(bert_input_ids, [[1, 2, 3, 0],
                                                 [2, 4, 3, 0],
                                                 [1, 2, 3, 0],
                                                 [2, 4, 3, 1],
                                                 [0, 0, 0, 0],
                                                 [2, 4, 3, 1],
                                                 [1, 3, 3, 1],
                                                 [1, 3, 3, 1],
                                                 [1, 2, 3, 0],
                                                 [2, 4, 3, 1],
                                                 [0, 0, 0, 0],
                                                 [20, 5, 3, 1],
                                                 [5, 6, 1, 1],
                                                 [5, 6, 1, 1]])
            # 1st query should be the same as 1nd query, 1st doc, 2nd field
            self.assertAllEqual(query_ftrs[0], doc_ftrs[0, 0, 1])
            # 1st query should be the same as 1nd query, 1st doc, 2nd field
            self.assertAllEqual(query_ftrs[0], doc_ftrs[0, 0, 0])

    def testBERTFtrExtWithUsrField(self):
        """test BERTFtrExt outputs and intermediate results"""

        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            usr_fields = [query, query, query]

            hparams = self.hparams

            bfe = bert_model.BertModel(
                query,
                doc_fields,
                usr_fields,
                hparams=hparams,
                mode=tf.estimator.ModeKeys.EVAL)

            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                query_ftrs, doc_ftrs, usr_ftrs, bert_input_ids = \
                    sess.run((bfe.query_ftrs, bfe.doc_ftrs, bfe.usr_ftrs, bfe.bert_input_ids))
                # check query/doc ftr shape
                self.assertAllEqual(query_ftrs.shape, [2, 100])
                self.assertAllEqual(doc_ftrs.shape, [2, 3, len(doc_fields), bfe.text_ftr_size])
                self.assertAllEqual(usr_ftrs.shape, [2, len(usr_fields), bfe.text_ftr_size])

                # 1st query should be the same as 1nd query, 1st doc, 2nd field
                self.assertAllEqual(query_ftrs[0], doc_ftrs[0, 0, 1])
                # 1st query should be the same as 1nd query, 1st doc, 2nd field
                self.assertAllEqual(query_ftrs[0], doc_ftrs[0, 0, 0])


if __name__ == "__main__":
    tf.test.main()
