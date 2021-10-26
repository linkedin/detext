import tensorflow as tf

from detext.layers import id_embed_layer
from detext.utils.parsing_utils import InputFtrType
from detext.utils.testing.data_setup import DataSetup


class TestIdEmbedLayer(tf.test.TestCase, DataSetup):
    """Unit test for id_embed_layer.py."""

    def testIdEmbedLayer(self):
        """test IdEmbedLayer outputs"""
        doc_id_field1 = tf.constant(self.ranking_doc_id_field1, dtype=tf.dtypes.string)
        doc_id_field2 = tf.constant(self.ranking_doc_id_field2, dtype=tf.dtypes.string)
        doc_id_fields = [doc_id_field1, doc_id_field2]

        user_id_field1 = tf.constant(self.user_id_field1, dtype=tf.dtypes.string)
        user_id_field2 = tf.constant(self.user_id_field2, dtype=tf.dtypes.string)
        user_id_fields = [user_id_field1, user_id_field2]

        id_embedder = id_embed_layer.IdEmbedLayer(
            num_id_fields=len(doc_id_fields) + len(user_id_fields),
            embedding_layer_param=self.embedding_layer_param,
            embedding_hub_url_for_id_ftr=''
        )

        id_ftr_size = id_embedder.id_ftr_size
        self.assertEqual(self.num_units_for_id_ftr, id_ftr_size)

        doc_ftrs, user_ftrs = id_embedder({InputFtrType.DOC_ID_COLUMN_NAMES: doc_id_fields, InputFtrType.USER_ID_COLUMN_NAMES: user_id_fields})
        self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, id_ftr_size])
        self.assertAllEqual(user_ftrs.shape, [2, 2, id_ftr_size])
        # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field (20, 5, 3, 1)
        self.assertAllEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
        # 1st query, 1st doc, 1st field should be the same as 1st query, 1st doc, 2nd field (1, 2, 3, 0)
        self.assertAllEqual(doc_ftrs[0, 0, 0], doc_ftrs[0, 0, 1])
        # For randomly chosed doc field (2nd sample, 3rd doc, 2nd field), vector should not be all zero because
        # initialized embedding should be non-zero
        self.assertNotAllClose(doc_ftrs[1, 2, 1], tf.zeros([self.num_units_for_id_ftr], dtype=tf.float32))


if __name__ == '__main__':
    tf.test.main()
