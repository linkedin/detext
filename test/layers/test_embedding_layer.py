import copy
import shutil

import tensorflow as tf
import tensorflow_hub as hub

from detext.layers import embedding_layer
from detext.utils.layer_utils import get_sorted_dict
from detext.utils.parsing_utils import InternalFtrType
from detext.utils.testing.data_setup import DataSetup


class TestEmbeddingLayer(tf.test.TestCase, DataSetup):
    """Tests embedding_layer.py"""
    num_cls_sep = 0
    min_len = 0
    max_len = 5

    sentences = tf.constant(['hello sent1', 'build sent2'])
    inputs = get_sorted_dict({InternalFtrType.SENTENCES: sentences,
                              InternalFtrType.NUM_CLS: tf.constant(num_cls_sep, dtype=tf.dtypes.int32),
                              InternalFtrType.NUM_SEP: tf.constant(num_cls_sep, dtype=tf.dtypes.int32),
                              InternalFtrType.MIN_LEN: tf.constant(min_len, dtype=tf.dtypes.int32),
                              InternalFtrType.MAX_LEN: tf.constant(max_len, dtype=tf.dtypes.int32)})

    embedding_layer_param = {'vocab_layer_param': DataSetup.vocab_layer_param,
                             'vocab_hub_url': '',
                             'we_file': '',
                             'we_trainable': True,
                             'num_units': DataSetup.num_units}

    def testEmbeddingLayerApi(self):
        """Checks whether a given layer conforms to the detext embedding layer api"""
        layer = hub.load(self.embedding_hub_url)
        layer: embedding_layer.EmbeddingLayerBase

        self.assertEqual(layer.num_units(), self.num_units)
        self.assertEqual(layer.vocab_size(), self.vocab_size)

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

    def testCreateEmbeddingLayer(self):
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

        loaded_layer = embedding_layer.create_embedding_layer(embedding_layer_param, embedding_hub_url=self.embedding_layer_dir)
        loaded_layer_outputs = loaded_layer(self.inputs)

        for k, v in outputs.items():
            self.assertAllEqual(v, loaded_layer_outputs[k])
        shutil.rmtree(self.embedding_layer_dir)


if __name__ == '__main__':
    tf.test.main()
