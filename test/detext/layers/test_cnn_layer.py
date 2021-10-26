import tensorflow as tf

from detext.layers import cnn_layer
from detext.utils.parsing_utils import InputFtrType
from detext.utils.testing.data_setup import DataSetup


class TestCnnLayer(tf.test.TestCase, DataSetup):
    """Unit test for cnn_layer.py."""
    num_filters = 20
    filter_window_sizes = [2]

    min_len = 3
    max_len = 4

    def testCnnLayer(self):
        """Test CNN layer """
        for embedding_hub_url in ['', self.embedding_hub_url]:
            self._testCnnLayer(embedding_hub_url)

    def _testCnnLayer(self, embedding_hub_url):
        query = self.query

        doc_fields = [self.ranking_doc_field1, self.ranking_doc_field2]
        user_fields = [query, query, query]

        num_filters = self.num_filters
        filter_window_sizes = self.filter_window_sizes
        layer = cnn_layer.CnnLayer(filter_window_sizes=filter_window_sizes,
                                   num_filters=num_filters, num_doc_fields=2, num_user_fields=3,
                                   min_len=self.min_len, max_len=self.max_len,
                                   embedding_layer_param=self.embedding_layer_param, embedding_hub_url=embedding_hub_url)
        text_ftr_size = num_filters * len(filter_window_sizes)

        query_ftrs, doc_ftrs, user_ftrs = layer(
            {InputFtrType.QUERY_COLUMN_NAME: query, InputFtrType.DOC_TEXT_COLUMN_NAMES: doc_fields, InputFtrType.USER_TEXT_COLUMN_NAMES: user_fields})
        self.assertEqual(text_ftr_size, layer.text_ftr_size)
        self.assertAllEqual(query_ftrs.shape, [2, text_ftr_size])
        self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, text_ftr_size])
        self.assertAllEqual(user_ftrs.shape, [2, 3, text_ftr_size])
        # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field (20, 5, 3, 1)
        self.assertAllEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
        # 1st query, 1st doc, 1st field should NOT be the same as 1st query, 1st doc, 2nd field (1, 2, 3, 0)
        self.assertNotAllClose(doc_ftrs[0, 0, 0], doc_ftrs[0, 0, 1])

    def testCnnConsistency(self):
        """ Test CNN consistency for data that only differ in batch sizes and padding tokens """
        # doc_field1 = tf.constant(self.doc_field1, dtype=tf.int32)
        doc_field1 = tf.constant(self.ranking_doc_field1, dtype=tf.dtypes.string)
        doc_fields = [doc_field1]
        user_fields = None

        filter_window_sizes = [3]
        num_filters = self.num_filters

        layer = cnn_layer.CnnLayer(filter_window_sizes=filter_window_sizes, num_units=self.num_units,
                                   num_filters=num_filters, num_doc_fields=1, num_user_fields=0,
                                   min_len=self.min_len, max_len=self.max_len,
                                   embedding_layer_param=self.embedding_layer_param, embedding_hub_url=None)
        query = ['batch1 query 1']
        query_ftrs, _, _ = layer(
            {InputFtrType.QUERY_COLUMN_NAME: query, InputFtrType.DOC_TEXT_COLUMN_NAMES: doc_fields, InputFtrType.USER_TEXT_COLUMN_NAMES: user_fields})

        query2 = query + ['batch 2 query build']
        query_ftrs2, _, _ = layer(
            {InputFtrType.QUERY_COLUMN_NAME: query2, InputFtrType.DOC_TEXT_COLUMN_NAMES: doc_fields, InputFtrType.USER_TEXT_COLUMN_NAMES: user_fields})
        self.assertAllClose(query_ftrs[0], query_ftrs2[0])


if __name__ == "__main__":
    tf.test.main()
