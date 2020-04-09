import tensorflow as tf

from detext.model import cnn_model


class TestCNNModel(tf.test.TestCase):
    """Unit test for cnn_model.py."""
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

    def testCNNModel(self):
        """test CNNFtrExt outputs"""

        with tf.Graph().as_default():
            query = tf.constant(self.query, dtype=tf.int32)
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_field2 = tf.constant(self.doc_field2, dtype=tf.int32)
            doc_fields = [doc_field1, doc_field2]
            usr_fields = [query, query, query]

            hparams = tf.contrib.training.HParams(
                we_trainable=True,
                use_tfr_loss=False,
                elem_rescale=False,
                filter_window_sizes=[2],
                num_units=100,
                num_filters=100,
                num_doc_fields=2,
                num_wide=5,
                vocab_size=30000,
                we_file=None,
                bert_checkpoint=None,
                use_wide=False,
                use_deep=True,
                num_hidden=[0],
                pad_id=0,
                explicit_empty=True
            )
            cnn = cnn_model.CnnModel(query,
                                     doc_fields,
                                     usr_fields,
                                     hparams=hparams,
                                     mode=tf.estimator.ModeKeys.EVAL)
            text_ftr_size = hparams.num_filters * len(hparams.filter_window_sizes)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                query_ftrs, doc_ftrs, usr_ftrs = sess.run((cnn.query_ftrs, cnn.doc_ftrs, cnn.usr_ftrs))
                self.assertEquals(text_ftr_size, cnn.text_ftr_size)
                self.assertAllEqual(query_ftrs.shape, [2, text_ftr_size])
                self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, text_ftr_size])
                self.assertAllEqual(usr_ftrs.shape, [2, 3, text_ftr_size])
                # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field (20, 5, 3, 1)
                self.assertAllEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
                # 1st query, 1st doc, 1st field should NOT be the same as 1st query, 1st doc, 2nd field (1, 2, 3, 0)
                self.assertNotAllClose(doc_ftrs[0, 0, 0], doc_ftrs[0, 0, 1])
                # 2nd query, 3rd doc, 2nd field should be all 0 because non-padding word number == filter_window_size
                self.assertAllEqual(doc_ftrs[1, 2, 1], tf.zeros([hparams.num_filters], dtype=tf.float32))

    def testCNNConsistency(self):
        """test CNN consistency in different batch"""
        with tf.Graph().as_default():
            query = tf.placeholder(dtype=tf.int32, shape=[None, None])
            doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
            doc_fields = [doc_field1]
            usr_fields = None

            hparams = tf.contrib.training.HParams(
                we_trainable=True,
                use_tfr_loss=False,
                elem_rescale=False,
                filter_window_sizes=[3],
                num_units=100,
                num_filters=100,
                num_doc_fields=1,
                vocab_size=30000,
                we_file=None,
                bert_checkpoint=None,
                use_wide=False,
                use_deep=True,
                num_hidden=[0],
                pad_id=0,
                explicit_empty=False
            )
            cnn = cnn_model.CnnModel(query,
                                     doc_fields,
                                     usr_fields,
                                     hparams,
                                     tf.estimator.ModeKeys.EVAL)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                query_ftrs = sess.run(cnn.query_ftrs, feed_dict={
                    query: [[1, 2, 3, 4]]})
                query_ftrs2 = sess.run(cnn.query_ftrs, feed_dict={
                    query: [[1, 2, 3, 4, 0],
                            [4, 5, 1, 2, 4]]})
                # 1st query ftrs should be the same as 2nd query ftrs
                self.assertAllClose(query_ftrs[0], query_ftrs2[0])


if __name__ == "__main__":
    tf.test.main()
