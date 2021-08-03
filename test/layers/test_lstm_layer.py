import tensorflow as tf

from detext.layers import lstm_layer
from detext.utils.parsing_utils import InputFtrType, InternalFtrType
from detext.utils.testing.data_setup import DataSetup


class TestLstmLayer(tf.test.TestCase, DataSetup):
    """Unit test for lstm_layer.py."""

    def testLstm(self):
        """Tests LSTM text encoder """
        for num_layers in [1, 2]:
            for bidirectional in [False, True]:
                self._testLstm(num_layers, bidirectional, '')

        for embedding_hub_url in ['', self.embedding_hub_url]:
            self._testLstm(1, True, embedding_hub_url)

    def _testLstm(self, num_layers, bidirectional, embedding_hub_url):
        """Tests LSTM text encoder given input """
        query = tf.constant(self.query, dtype=tf.dtypes.string)
        doc_field1 = tf.constant(self.ranking_doc_field1, dtype=tf.dtypes.string)
        doc_field2 = tf.constant(self.ranking_doc_field2, dtype=tf.dtypes.string)
        doc_fields = [doc_field1, doc_field2]
        user_fields = [query, query, query]
        num_units = self.num_units

        layer = lstm_layer.LstmLayer(
            we_trainable=True,
            num_units=self.num_units,
            num_doc_fields=2,
            num_layers=num_layers,
            forget_bias=0.5,
            rnn_dropout=0.,
            bidirectional=bidirectional,
            min_len=self.min_len, max_len=self.max_len,
            embedding_layer_param=self.embedding_layer_param, embedding_hub_url=embedding_hub_url
        )
        text_ftr_size = num_units

        query_ftrs, doc_ftrs, user_ftrs = layer(
            {InputFtrType.QUERY_COLUMN_NAME: query, InputFtrType.DOC_TEXT_COLUMN_NAMES: doc_fields, InputFtrType.USER_TEXT_COLUMN_NAMES: user_fields},
            training=False)

        self.assertEqual(text_ftr_size, layer.text_ftr_size)
        self.assertAllEqual(query_ftrs.shape, [2, text_ftr_size])
        self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, text_ftr_size])
        self.assertAllEqual(user_ftrs.shape, [2, 3, text_ftr_size])

        # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field
        self.assertAllEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
        # 1st query, 1st doc, 1st field should NOT be the same as 1st query, 2nd doc, 1st field
        self.assertNotAllClose(doc_ftrs[0, 0, 0], doc_ftrs[0, 1, 0])

    def testApplyLstmOnText(self):
        """Tests apply_lstm_on_text() """
        for num_layers in [1, 2]:
            for bidirectional in [True]:
                self._testApplyLstmOnText(num_layers, bidirectional)

    def _testApplyLstmOnText(self, num_layers, bidirectional):
        """Tests apply_lstm_on_text() given input """
        query = tf.constant(self.query, dtype=tf.dtypes.string)
        doc_field1 = tf.constant(self.ranking_doc_field1, dtype=tf.dtypes.string)
        doc_field2 = tf.constant(self.ranking_doc_field2, dtype=tf.dtypes.string)
        doc_fields = [doc_field1, doc_field2]

        num_units = self.num_units
        layer = lstm_layer.LstmLayer(
            we_trainable=True,
            num_units=self.num_units,
            num_layers=num_layers,
            forget_bias=0.5,
            rnn_dropout=0.,
            bidirectional=bidirectional,
            min_len=self.min_len, max_len=self.max_len,
            embedding_layer_param=self.embedding_layer_param, embedding_hub_url=None
        )

        query_ftrs, doc_ftrs, user_ftrs = layer({InputFtrType.QUERY_COLUMN_NAME: query, InputFtrType.DOC_TEXT_COLUMN_NAMES: doc_fields}, training=False)
        results = lstm_layer.apply_lstm_on_text(query, layer.text_encoders, layer.embedding,
                                                bidirectional, self.min_len, self.max_len, layer.num_cls_sep,
                                                False)
        query_seq_outputs = results[InternalFtrType.SEQ_OUTPUTS]
        query_memory_state = results[InternalFtrType.LAST_MEMORY_STATE]

        # Make sure layer.call() and apply_lstm_on_text output the same result
        self.assertAllEqual(query_ftrs, query_memory_state)

        # Make sure sequence outputs are different at each token
        self.assertNotAllEqual(query_seq_outputs[0][2], query_seq_outputs[0][0])
        self.assertNotAllEqual(query_seq_outputs[0][2], query_seq_outputs[0][1])
        if not bidirectional:
            # Make sure output is the last state that's not masked out by the sequence mask inferred from sequence length
            expected = tf.stack([query_seq_outputs[0][2], query_seq_outputs[1][3]], axis=0)
            self.assertAllEqual(query_memory_state, expected)
        else:
            first_query_end = min(self.query_length[0] + 1, self.max_len - 1)
            second_query_end = min(self.query_length[1] + 1, self.max_len - 1)
            # In the bidirectional LSTM cases, the last state of the backward layer is the memory state of the **first** token.
            # Therefore, the checking above does not apply. Instead, we check the forward and backward output separately
            expected_fw_last_state = tf.stack([query_seq_outputs[0][first_query_end][:num_units // 2], query_seq_outputs[1][second_query_end][:num_units // 2]],
                                              axis=0)
            expected_bw_last_state = tf.stack([query_seq_outputs[0][0][num_units // 2:], query_seq_outputs[1][0][num_units // 2:]], axis=0)
            self.assertAllEqual(tf.slice(query_memory_state, [0, 0], [len(query), num_units // 2]), expected_fw_last_state)
            self.assertAllEqual(tf.slice(query_memory_state, [0, num_units // 2], [len(query), num_units // 2]), expected_bw_last_state)


if __name__ == "__main__":
    tf.test.main()
