import copy
import shutil

import tensorflow as tf
import tensorflow_hub as hub

from detext.layers import vocab_layer
from detext.utils.layer_utils import get_sorted_dict
from detext.utils.parsing_utils import InternalFtrType
from detext.utils.testing.data_setup import DataSetup


class TestVocabLayer(tf.test.TestCase, DataSetup):
    num_cls_sep = 1
    sentences = tf.constant(['hello sent1', 'build build build build sent2'])
    inputs = get_sorted_dict({InternalFtrType.SENTENCES: sentences,
                              InternalFtrType.NUM_CLS: tf.constant(num_cls_sep, dtype=tf.dtypes.int32),
                              InternalFtrType.NUM_SEP: tf.constant(num_cls_sep, dtype=tf.dtypes.int32),
                              InternalFtrType.MIN_LEN: tf.constant(DataSetup.min_len, dtype=tf.dtypes.int32),
                              InternalFtrType.MAX_LEN: tf.constant(DataSetup.max_len, dtype=tf.dtypes.int32)})

    def testAddClsSep(self):
        vocab_layer_param = copy.copy(self.vocab_layer_param)
        inputs = copy.copy(self.inputs)
        inputs['min_len'] = 6
        inputs['max_len'] = 7
        inputs['num_cls'] = 2
        inputs['num_sep'] = 2

        layer = vocab_layer.create_vocab_layer(vocab_layer_param, '')
        outputs = layer(inputs)

        self.assertAllEqual(outputs[InternalFtrType.TOKENIZED_IDS][0],
                            tf.constant([self.CLS_ID, self.CLS_ID, self.UNK_ID, self.UNK_ID, self.SEP_ID, self.SEP_ID, self.PAD_ID]))

    def testAdjustLen(self):
        vocab_layer_param = copy.copy(self.vocab_layer_param)
        inputs = copy.copy(self.inputs)
        inputs['min_len'] = 12
        inputs['max_len'] = 16

        layer = vocab_layer.create_vocab_layer(vocab_layer_param, '')
        outputs = layer(inputs)
        shape = tf.shape(outputs[InternalFtrType.TOKENIZED_IDS])
        self.assertAllEqual(shape, tf.constant([2, 12]))

        inputs['min_len'] = 0
        inputs['max_len'] = 1
        outputs = layer(inputs)
        shape = tf.shape(outputs[InternalFtrType.TOKENIZED_IDS])
        self.assertAllEqual(shape, tf.constant([2, 1]))

    def testLength(self):
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
        """Checks whether a given layer conforms to the DeText vocab layer API"""
        layer = hub.load(self.vocab_hub_url)
        layer: vocab_layer.VocabLayerBase

        self.assertEqual(layer.vocab_size(), self.vocab_size)
        self.assertEqual(layer.pad_id(), self.PAD_ID)

        inputs = self.inputs
        outputs = layer(inputs)
        expected_outputs = {InternalFtrType.LENGTH: tf.constant([4, 7]),
                            InternalFtrType.TOKENIZED_IDS: tf.constant([[1, 0, 0, 2, 3, 3, 3],
                                                                        [1, 4, 4, 4, 4, 0, 2]])}

        for k, v in outputs.items():
            self.assertAllEqual(v, expected_outputs[k])

    def testCreateVocabLayer(self):
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
