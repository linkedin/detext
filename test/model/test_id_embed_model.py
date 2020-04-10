import numpy as np
import tensorflow as tf

from detext.model import id_embed_model


class TestIdEmbedModel(tf.test.TestCase):
    """Unit test for id_embed_model.py."""
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

    def testIdEmbedModel(self):
        """test IdEmbedModel outputs"""

        with tf.Graph().as_default():
            doc_id_field1 = tf.constant(self.doc_id_field1, dtype=tf.int32)
            doc_id_field2 = tf.constant(self.doc_id_field2, dtype=tf.int32)
            doc_id_fields = [doc_id_field1, doc_id_field2]

            usr_id_field1 = tf.constant(self.usr_id_field1, dtype=tf.int32)
            usr_id_field2 = tf.constant(self.usr_id_field2, dtype=tf.int32)
            usr_id_fields = [usr_id_field1, usr_id_field2]

            hparams = tf.contrib.training.HParams(
                num_doc_id_fields=2,
                num_usr_id_fields=2,
                num_units_for_id_ftr=10,
                vocab_size_for_id_ftr=30,
                we_trainable_for_id_ftr=True,
                we_file_for_id_ftr=None,
                pad_id_for_id_ftr=0
            )
            id_embedder = id_embed_model.IdEmbedModel(
                doc_id_fields=doc_id_fields,
                usr_id_fields=usr_id_fields,
                hparams=hparams,
                mode=tf.estimator.ModeKeys.EVAL)

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                doc_ftrs, usr_ftrs = sess.run((id_embedder.doc_ftrs, id_embedder.usr_ftrs))
                self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, hparams.num_units_for_id_ftr])
                self.assertAllEqual(usr_ftrs.shape, [2, 2, hparams.num_units_for_id_ftr])
                # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field (20, 5, 3, 1)
                self.assertAllEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
                # 1st query, 1st doc, 1st field should be the same as 1st query, 1st doc, 2nd field (1, 2, 3, 0)
                self.assertAllEqual(doc_ftrs[0, 0, 0], doc_ftrs[0, 0, 1])
                # For randomly chosed doc field (2nd sample, 3rd doc, 2nd field), vector should not be all zero because
                # initialized embedding should be non-zero
                self.assertNotAllClose(doc_ftrs[1, 2, 1], tf.zeros([hparams.num_units_for_id_ftr], dtype=tf.float32))
                # Manually compute the average embedding and compare with model output
                # This is to make sure that padding embedding is not considered when computing average embedding vector
                data = []
                embedding = id_embedder.embedding.eval()
                for wid in doc_id_field2.eval()[1][2]:
                    if wid != hparams.pad_id_for_id_ftr:
                        data.append(embedding[wid])
                self.assertAllEqual(doc_ftrs[1, 2, 1], np.mean(data, axis=0))
