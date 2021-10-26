import copy
import shutil

import tensorflow as tf
import tensorflow_hub as hub

from smart_compose.layers import embedding_layer
from smart_compose.utils.layer_utils import get_sorted_dict
from smart_compose.utils.parsing_utils import InternalFtrType
from smart_compose.utils.testing.data_setup import DataSetup
from smart_compose.utils.testing.test_case import TestCase


class TestEmbeddingLayer(TestCase):
    """Tests embedding_layer.py"""
    num_cls = 0
    num_sep = 0

    min_len = 0
    max_len = 5

    sentences = tf.constant(['hello sent1', 'build sent2'])
    inputs = get_sorted_dict({'sentences': sentences,
                              'num_cls': tf.constant(num_cls, dtype=tf.dtypes.int32),
                              'num_sep': tf.constant(num_sep, dtype=tf.dtypes.int32),
                              'min_len': tf.constant(min_len, dtype=tf.dtypes.int32),
                              'max_len': tf.constant(max_len, dtype=tf.dtypes.int32)})

    embedding_layer_param = {'vocab_layer_param': DataSetup.vocab_layer_param,
                             'vocab_hub_url': '',
                             'we_file': '',
                             'we_trainable': True,
                             'num_units': DataSetup.num_units}

    def testCreateEmbeddingLayer(self):
        """Tests create_embedding_layer() """
        for vocab_hub_url in ['', self.vocab_hub_url]:
            embedding_layer_param = copy.copy(self.embedding_layer_param)
            embedding_layer_param['vocab_hub_url'] = vocab_hub_url
            self._testCreateEmbeddingLayer('', embedding_layer_param)

        embedding_layer_param = copy.copy(self.embedding_layer_param)
        embedding_layer_param['we_file'] = self.we_file
        self._testCreateEmbeddingLayer('', embedding_layer_param)

        embedding_layer_param = copy.copy(self.embedding_layer_param)
        self._testCreateEmbeddingLayer(self.embedding_hub_url, embedding_layer_param)

    def _testCreateEmbeddingLayer(self, embedding_hub_url, embedding_layer_param):
        layer = embedding_layer.create_embedding_layer(embedding_layer_param, embedding_hub_url)
        outputs = layer(self.inputs)

        tf.saved_model.save(layer, self.embedding_layer_dir)

        loaded_layer = embedding_layer.create_embedding_layer(embedding_layer_param, self.embedding_layer_dir)
        loaded_layer_outputs = loaded_layer(self.inputs)

        for k, v in outputs.items():
            self.assertAllEqual(v, loaded_layer_outputs[k])

        shutil.rmtree(self.embedding_layer_dir)

    def testEmbeddingLayerApi(self):
        """Checks whether a given layer conforms to the smart compose embedding layer api"""
        layer = hub.load(self.embedding_hub_url)
        layer: embedding_layer.EmbeddingLayerBase

        self.assertEqual(layer.num_units(), self.num_units)
        self.assertEqual(layer.vocab_size(), self.vocab_size)
        self.assertEqual(layer.sep_id(), self.SEP_ID)

        tokenized = layer.tokenize_to_indices(self.inputs)
        expected_tokenized = {InternalFtrType.LENGTH: tf.constant([2, 2]),
                              InternalFtrType.TOKENIZED_IDS: tf.constant([[0, 0],
                                                                          [4, 0]])}
        for k, v in tokenized.items():
            self.assertAllEqual(v, expected_tokenized[k])

        tokenized_result = tf.constant([[1, 2], [0, 1]])
        tokenized_result_shape = tf.shape(tokenized_result)
        embedding_lookup_result = layer.embedding_lookup(tokenized_result)
        self.assertAllEqual(tf.shape(embedding_lookup_result), [tokenized_result_shape[0], tokenized_result_shape[1], layer.num_units()])

        outputs = layer(self.inputs)
        self.assertAllEqual(tf.shape(outputs[InternalFtrType.EMBEDDED]), [tokenized_result_shape[0], tokenized_result_shape[1], layer.num_units()])
        self.assertAllEqual(outputs[InternalFtrType.LENGTH], tf.constant([2, 2]))


if __name__ == '__main__':
    tf.test.main()
