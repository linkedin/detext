import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text
from libert.preprocess import create_tf_vocab_from_sp_tokenizer

from detext.layers import bert_layer
from detext.utils.parsing_utils import InputFtrType
from detext.utils.testing.data_setup import DataSetup
from detext.utils.vocab_utils import read_vocab


class TestBertLayer(tf.test.TestCase, DataSetup):
    """Unit test for bert_layer.py"""
    # Bert setup
    bert_hub_layer = hub.KerasLayer(hub.resolve(DataSetup.bert_hub_url), trainable=True)

    bert_vocab_file = bert_hub_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
    bert_vocab_table = read_vocab(bert_vocab_file)

    bert_PAD_ID = bert_vocab_table[DataSetup.PAD]
    bert_SEP_ID = bert_vocab_table[DataSetup.SEP]
    bert_CLS_ID = bert_vocab_table[DataSetup.CLS]
    bert_UNK_ID = bert_vocab_table[DataSetup.UNK]

    # SentencePiece setup
    sentencepiece_hub_layer = hub.KerasLayer(hub.resolve(DataSetup.libert_sp_hub_url))

    tokenizer_file = sentencepiece_hub_layer.resolved_object.tokenizer_file.asset_path.numpy().decode("utf-8")
    with tf.io.gfile.GFile(tokenizer_file, 'rb') as f_handler:
        sp_model = f_handler.read()

    sentencepiece_tokenizer = tf_text.SentencepieceTokenizer(model=sp_model, out_type=tf.int32)
    sentencepiece_vocab_tf_table = create_tf_vocab_from_sp_tokenizer(sp_tokenizer=sentencepiece_tokenizer, num_oov_buckets=1)

    sentencepiece_PAD_ID = sentencepiece_vocab_tf_table.lookup(tf.constant(DataSetup.PAD)).numpy()
    sentencepiece_SEP_ID = sentencepiece_vocab_tf_table.lookup(tf.constant(DataSetup.SEP)).numpy()
    sentencepiece_CLS_ID = sentencepiece_vocab_tf_table.lookup(tf.constant(DataSetup.CLS)).numpy()
    sentencepiece_UNK_ID = sentencepiece_vocab_tf_table.lookup(tf.constant(DataSetup.UNK)).numpy()

    # Space setup
    space_hub_layer = hub.KerasLayer(hub.resolve(DataSetup.libert_space_hub_url))

    tokenizer_file = space_hub_layer.resolved_object.tokenizer_file.asset_path.numpy().decode("utf-8")
    space_vocab = read_vocab(tokenizer_file)

    space_PAD_ID = space_vocab[DataSetup.PAD]
    space_SEP_ID = space_vocab[DataSetup.SEP]
    space_CLS_ID = space_vocab[DataSetup.CLS]
    space_UNK_ID = space_vocab[DataSetup.UNK]

    # Hyperparameters setup
    num_units = 16
    pad_id = 0
    num_doc_fields = 2

    min_len = 3
    max_len = 8

    layer = bert_layer.BertLayer(num_units, DataSetup.CLS, DataSetup.SEP, DataSetup.PAD, DataSetup.UNK, min_len, max_len, DataSetup.bert_hub_url)

    def testBertLayer(self):
        """Test Bert layer """
        for hub_url in [self.bert_hub_url]:
            self._testBertLayer(hub_url)

    def _testBertLayer(self, hub_url):
        query = self.query

        doc_fields = [self.ranking_doc_field1, self.ranking_doc_field2]
        user_fields = [query, query, query]

        query_ftrs, doc_ftrs, user_ftrs = self.layer(
            {InputFtrType.QUERY_COLUMN_NAME: query, InputFtrType.DOC_TEXT_COLUMN_NAMES: doc_fields, InputFtrType.USER_TEXT_COLUMN_NAMES: user_fields},
            False)

        text_ftr_size = self.num_units

        self.assertEqual(text_ftr_size, self.layer.text_ftr_size)
        self.assertAllEqual(query_ftrs.shape, [2, self.layer.text_ftr_size])
        self.assertAllEqual(doc_ftrs.shape, [2, 3, 2, self.layer.text_ftr_size])
        self.assertAllEqual(user_ftrs.shape, [2, 3, self.layer.text_ftr_size])
        # 1st query, 2nd doc, 2nd field should be the same as 2nd query, 1st doc, 2nd field
        self.assertAllEqual(doc_ftrs[0, 1, 1], doc_ftrs[1, 0, 1])
        # 1st query, 1st doc, 1st field should be the same as 1st query, 1st doc, 2nd field
        self.assertAllEqual(doc_ftrs[0, 0, 0], doc_ftrs[0, 0, 1])
        # 1st query, 1st doc, 2st field should NOT be the same as 1st query, 2st doc, 2nd field
        self.assertNotAllClose(doc_ftrs[0, 1, 0], doc_ftrs[0, 1, 1])

    def testGetInputIds(self):
        """Tests get_input_ids() """
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
        query = tf.constant(query, dtype=tf.int32)
        doc_field1 = tf.constant(doc_field1, dtype=tf.int32)
        doc_field2 = tf.constant(doc_field2, dtype=tf.int32)
        doc_fields = [doc_field1, doc_field2]
        user_fields = None

        max_text_len, max_text_len_array = bert_layer.BertLayer.get_input_max_len(query, doc_fields, user_fields)
        bert_input_ids = bert_layer.BertLayer.get_bert_input_ids(query, doc_fields, user_fields, self.pad_id, max_text_len, max_text_len_array)

        # Check bert input ids
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

    def testPrerocessQuery(self):
        """Tests _preprocess_query function of bert layer"""
        query = tf.constant(['batch 1 user 1 build',
                             'batch 2 user 2 word'], dtype=tf.string)

        expected = tf.constant([[self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 4, 2],
                                [self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 5, 2]],
                               dtype=tf.int32)

        self.assertAllEqual(expected, self.layer._preprocess_query(query))

    def testPrerocessUsr(self):
        """Tests _preprocess_user function of bert layer"""
        user_fields = [tf.constant(['batch 1 user 1 build',
                                    'batch 2 user 2 word'], dtype=tf.string)]

        expected = [tf.constant([[self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 4, 2],
                                 [self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 5, 2]],
                                dtype=tf.int32)]

        self.assertAllEqual(expected, self.layer._preprocess_user(user_fields))

    def testPrerocessDoc(self):
        """Tests _preprocess_doc function of bert layer"""
        doc_fields = [tf.constant([['batch 1 doc 1 build', 'batch 1 doc 2'],
                                   ['batch 2 doc 1 word', 'batch 2 doc 2']], dtype=tf.string)]

        expected = [tf.constant([[[self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 4, 2],
                                  [self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 2, 3]],

                                 [[self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 5, 2],
                                  [self.CLS_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, self.bert_UNK_ID, 2, 3]]],
                                dtype=tf.int32)]

        self.assertAllEqual(expected, self.layer._preprocess_doc(doc_fields))

    def testBertPreprocessLayerWordPiece(self):
        """Tests BertPreprocessLayer with wordpiece tokenizer"""

        preprocess_layer = bert_layer.BertPreprocessLayer(self.bert_hub_layer, self.max_len, self.min_len, self.CLS, self.SEP, self.PAD, self.UNK)

        sentences = tf.constant(['test sent1', 'build build build build sent2'])

        expected = tf.constant([[self.bert_CLS_ID, 8, self.bert_UNK_ID, self.bert_SEP_ID, self.bert_PAD_ID, self.bert_PAD_ID, self.bert_PAD_ID],
                                [self.bert_CLS_ID, 4, 4, 4, 4, self.bert_UNK_ID, self.bert_SEP_ID]],
                               dtype=tf.int32)
        outputs = preprocess_layer(sentences)
        self.assertAllEqual(expected, outputs)

    def testBertPreprocessLayerSentencePiece(self):
        """Tests BertPreprocessLayer with sentencepiece tokenizer"""

        preprocess_layer = bert_layer.BertPreprocessLayer(self.sentencepiece_hub_layer, self.max_len, self.min_len, self.CLS, self.SEP, self.PAD, self.UNK)

        sentences = tf.constant(['TEST sent1', 'build build build build sent2'])

        expected = tf.constant([[self.sentencepiece_CLS_ID, 557, 4120, 29900, self.sentencepiece_SEP_ID, self.sentencepiece_PAD_ID,
                                 self.sentencepiece_PAD_ID, self.sentencepiece_PAD_ID],
                                [self.sentencepiece_CLS_ID, 671, 671, 671, 671, 4120, 29904, self.sentencepiece_SEP_ID]],
                               dtype=tf.int32)

        outputs = preprocess_layer(sentences)
        self.assertAllEqual(expected, outputs)

    def testBertPreprocessLayerSpace(self):
        """Tests BertPreprocessLayer with space tokenizer"""

        preprocess_layer = bert_layer.BertPreprocessLayer(self.space_hub_layer, self.max_len, self.min_len, self.CLS, self.SEP, self.PAD, self.UNK)

        sentences = tf.constant(['test sent1', 'build build build build sent2'])

        expected = tf.constant([[self.space_CLS_ID, 8, self.space_UNK_ID, self.space_SEP_ID, self.space_PAD_ID, self.space_PAD_ID, self.space_PAD_ID],
                                [self.space_CLS_ID, 4, 4, 4, 4, self.space_UNK_ID, self.space_SEP_ID]],
                               dtype=tf.int32)

        outputs = preprocess_layer(sentences)
        self.assertAllEqual(expected, outputs)

    def testBertPreprocessLayerAdjustLen(self):
        """Tests adjust_len function of BertPreprocessLayer"""

        sentences = tf.constant(['test sent1', 'build build build build sent2'])

        min_len = 12
        max_len = 16

        layer = bert_layer.BertPreprocessLayer(self.bert_hub_layer, max_len, min_len, self.CLS, self.SEP, self.PAD, self.UNK)

        outputs = layer(sentences)
        shape = tf.shape(outputs)

        self.assertAllEqual(shape, tf.constant([2, 12]))

        min_len = 0
        max_len = 1

        layer = bert_layer.BertPreprocessLayer(self.bert_hub_layer, max_len, min_len, self.CLS, self.SEP, self.PAD, self.UNK)

        outputs = layer(sentences)
        shape = tf.shape(outputs)

        self.assertAllEqual(shape, tf.constant([2, 1]))


if __name__ == "__main__":
    tf.test.main()
