import copy
import shutil

import tensorflow as tf
import tensorflow_hub as hub

from smart_compose.layers import vocab_layer
from smart_compose.utils.layer_utils import get_sorted_dict
from smart_compose.utils.parsing_utils import InternalFtrType
from smart_compose.utils.testing.data_setup import DataSetup
from smart_compose.utils.testing.test_case import TestCase


class TestVocabLayer(TestCase):
    """Tests vocab_layer.py """
    num_cls = 1
    num_sep = 1
    sentences = tf.constant(['hello sent1', 'build build build build sent2'])
    inputs = get_sorted_dict({'sentences': sentences,
                              'num_cls': tf.constant(num_cls, dtype=tf.dtypes.int32),
                              'num_sep': tf.constant(num_sep, dtype=tf.dtypes.int32),
                              'min_len': tf.constant(DataSetup.min_len, dtype=tf.dtypes.int32),
                              'max_len': tf.constant(DataSetup.max_len, dtype=tf.dtypes.int32)})
    layer = vocab_layer.create_vocab_layer(TestCase.vocab_layer_param, '')

    def testAddClsSep(self):
        """Tests add_cls_sep() """
        inputs = copy.copy(self.inputs)
        inputs['min_len'] = 6
        inputs['max_len'] = 7

        num_cls_lst = [1, 0, 1]
        num_sep_lst = [2, 2, 0]

        expected_output_lst = [
            tf.constant([self.CLS_ID, self.UNK_ID, self.UNK_ID, self.SEP_ID, self.SEP_ID, self.PAD_ID, self.PAD_ID]),
            tf.constant([self.UNK_ID, self.UNK_ID, self.SEP_ID, self.SEP_ID, self.PAD_ID, self.PAD_ID, self.PAD_ID]),
            tf.constant([self.CLS_ID, self.UNK_ID, self.UNK_ID, self.PAD_ID, self.PAD_ID, self.PAD_ID]),
        ]

        for num_cls, num_sep, expected_output in zip(num_cls_lst, num_sep_lst, expected_output_lst):
            inputs['num_cls'] = num_cls
            inputs['num_sep'] = num_sep
            self._testAddClsSep(inputs, self.layer, expected_output)

    def _testAddClsSep(self, inputs, layer, expected_output):
        outputs = layer(inputs)
        self.assertAllEqual(outputs[InternalFtrType.TOKENIZED_IDS][0], expected_output)

    def testAdjustLen(self):
        """Tests adjust_len() """
        inputs = copy.copy(self.inputs)
        inputs['min_len'] = 12
        inputs['max_len'] = 16

        outputs = self.layer(inputs)
        shape = tf.shape(outputs[InternalFtrType.TOKENIZED_IDS])
        self.assertAllEqual(shape, tf.constant([2, 12]))

        inputs['min_len'] = 0
        inputs['max_len'] = 1
        outputs = self.layer(inputs)
        shape = tf.shape(outputs[InternalFtrType.TOKENIZED_IDS])
        self.assertAllEqual(shape, tf.constant([2, 1]))

    def testAdjustLenRight(self):
        """Tests adjust_len_right() """
        inputs = tf.convert_to_tensor([[2, 1]], dtype=tf.int32)
        min_len = tf.convert_to_tensor(1, dtype=tf.int32)
        max_len = tf.convert_to_tensor(1, dtype=tf.int32)
        outputs = self.layer.adjust_len_right(inputs, min_len, max_len)
        self.assertAllEqual(outputs, tf.convert_to_tensor([[1]]))

        inputs = tf.convert_to_tensor([[2, 1]], dtype=tf.int32)
        min_len = tf.convert_to_tensor(1, dtype=tf.int32)
        max_len = tf.convert_to_tensor(3, dtype=tf.int32)
        outputs = self.layer.adjust_len_right(inputs, min_len, max_len)
        self.assertAllEqual(outputs, tf.convert_to_tensor([[2, 1]]))

    def testKeys(self):
        """Tests keys()"""
        self.assertAllEqual(self.layer.keys(),
                            [b'[UNK]',
                             b'[CLS]',
                             b'[SEP]',
                             b'[PAD]',
                             b'build',
                             b'word',
                             b'function',
                             b'able',
                             b'test',
                             b'this',
                             b'is',
                             b'a',
                             b'source',
                             b'sentence',
                             b'target',
                             b'token'])

    def testValues(self):
        """Tests values()"""
        self.assertAllEqual(
            self.layer.values(), list(range(len(self.layer.keys())))
        )

    def testLength(self):
        """Tests length()"""
        vocab_layer_param = copy.copy(self.vocab_layer_param)
        inputs = copy.copy(self.inputs)
        inputs['min_len'] = 1
        inputs['max_len'] = 16
        inputs['num_cls'] = 0
        inputs['num_sep'] = 0

        layer = vocab_layer.create_vocab_layer(vocab_layer_param, '')
        outputs = layer(inputs)
        self.assertAllEqual(outputs[InternalFtrType.LENGTH], tf.constant([2, 5]))

        inputs['num_cls'] = 1
        inputs['num_sep'] = 1
        layer = vocab_layer.create_vocab_layer(vocab_layer_param, '')
        outputs = layer(inputs)
        self.assertAllEqual(outputs[InternalFtrType.LENGTH], tf.constant([4, 7]))

    def testVocabLayerApi(self):
        """Checks whether a given layer conforms to the smart compose vocab layer API"""
        layer = hub.load(self.vocab_hub_url)
        layer: vocab_layer.VocabLayerBase

        self.assertEqual(layer.vocab_size(), self.vocab_size)
        self.assertEqual(layer.pad_id(), self.PAD_ID)
        self.assertEqual(layer.sep_id(), self.SEP_ID)

    def testVocabLookup(self):
        """Tests vocab_lookup()"""
        vocab_layer_param = copy.copy(self.vocab_layer_param)
        layer = vocab_layer.create_vocab_layer(vocab_layer_param, '')
        outputs = layer.vocab_lookup(tf.sparse.from_dense([["hello", "build"]]))
        self.assertAllEqual(
            tf.sparse.to_dense(outputs), tf.convert_to_tensor([[0, 4]])
        )

    def testConvertIdsToTexts(self):
        """Tests convert_ids_to_texts()"""
        vocab_layer_param = copy.copy(self.vocab_layer_param)
        layer = vocab_layer.create_vocab_layer(vocab_layer_param, '')
        inputs = self.inputs
        outputs = layer(inputs)
        expected_tokenized_result = tf.constant([[1, 0, 0, 2, 3, 3, 3],
                                                 [1, 4, 4, 4, 4, 0, 2]])
        expected_outputs = {InternalFtrType.LENGTH: tf.constant([4, 7]),
                            InternalFtrType.TOKENIZED_IDS: expected_tokenized_result}

        for k, v in outputs.items():
            self.assertAllEqual(v, expected_outputs[k])

        expected_inverse_vocab_lookup_results = [b'[CLS] [UNK] [UNK] [SEP] [PAD] [PAD] [PAD]', b'[CLS] build build build build [UNK] [SEP]']
        self.assertAllEqual(layer.convert_ids_to_texts(expected_tokenized_result), expected_inverse_vocab_lookup_results)

    def testCreateVocabLayer(self):
        """Tests create_vocab_layer() """
        for vocab_hub_url in ['', self.vocab_hub_url]:
            self._testCreateVocabLayer(vocab_hub_url)

    def _testCreateVocabLayer(self, vocab_hub_url):
        layer = vocab_layer.create_vocab_layer(self.vocab_layer_param, vocab_hub_url)
        outputs = layer(self.inputs)
        tf.saved_model.save(layer, self.vocab_layer_dir)

        loaded_layer = vocab_layer.create_vocab_layer(None, self.vocab_layer_dir)
        loaded_layer_outputs = loaded_layer(self.inputs)

        for k, v in outputs.items():
            self.assertAllEqual(v, loaded_layer_outputs[k])

        shutil.rmtree(self.vocab_layer_dir)


if __name__ == '__main__':
    tf.test.main()
