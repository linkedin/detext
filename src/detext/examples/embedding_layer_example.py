import pickle
import sys
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_hub as hub
from absl import logging
from smart_arg import arg_suite

from detext.layers import embedding_layer
from detext.utils.layer_utils import get_sorted_dict
from detext.utils.parsing_utils import InternalFtrType


@arg_suite
@dataclass
class Args:
    vocab_hub_url: str  # TF hub URL to vocab layer
    embedding_file: str  # Embedding matrix in pickle format. Shape=[vocab_size, num_units]
    num_units: int  # Dimension of embedding file
    trainable: bool  # Whether embedding is trainable
    output_file: str  # Output location of embedding layer


def init_word_embedding(vocab_size, num_units, we_trainable, we_file=None, name_prefix="w"):
    """Initialize word embeddings from random initialization or pretrained word embedding.

    This function is only used by encoding models other than BERT
    """

    if not we_file:
        embedding_name = "{}_pretrained_embedding".format(name_prefix)
        # Random initialization
        embedding = tf.compat.v1.get_variable(
            embedding_name, [vocab_size, num_units], dtype=tf.float32, trainable=we_trainable)
    else:
        # Initialize by pretrained word embedding
        embedding_name = "{}_embedding".format(name_prefix)
        we = pickle.load(tf.io.gfile.GFile(we_file, 'rb'))
        assert vocab_size == we.shape[0] and num_units == we.shape[1]
        embedding = tf.compat.v1.get_variable(name=embedding_name,
                                              shape=[vocab_size, num_units],
                                              dtype=tf.float32,
                                              initializer=tf.compat.v1.constant_initializer(we),
                                              trainable=we_trainable
                                              )
    return embedding


class ExampleEmbeddingLayer(embedding_layer.EmbeddingLayerBase):
    """Example embedding layer accepted by DeText

    To check whether a given vocab layer conforms to the DeText API, follow the test function in detext unit tests:
      test/layers/test_embedding_layer.testEmbeddingLayerApi()
    """

    def __init__(self, vocab_hub_url, we_file, we_trainable, num_units, name_prefix='w'):
        """ Initializes the embedding layer

        :param vocab_hub_url Url to saved vocabulary layer. If empty string or None, no vocab layer will be loaded
        :param we_file Path to pretrained word embedding
        :param we_trainable Whether word embedding is trainable
        :param num_units Dimension of embedding
        :param name_prefix Prefix of embedding variables
        """
        super().__init__()
        self.vocab_layer = hub.load(vocab_hub_url)  # A vocab layer accepted by DeText. Check example/vocab_layer_example for an example
        self._num_units = num_units
        self._vocab_size = self.vocab_layer.vocab_size()

        self.embedding = init_word_embedding(self._vocab_size, num_units, we_trainable, we_file, name_prefix)

    @tf.function
    def tokenize_to_indices(self, inputs):
        """Tokenize given inputs and convert to indices

        Example: tokenize_to_indices(['hello world', 'sentence 1 token']) -> {RESULT: [[20, 10, pad_id], [4, 5, 6]], LENGTH: [2, 3]}

        :param inputs tf.Tensor(dtype=string) Shape=[batch_size]
        :return A dictionary containing the following key values:
          RESULT: tf.Tensor(dtype=int) Shape=[batch_size, sentence_len]. Tokenization and lookup result
          LENGTH: tf.Tensor(dtype=int) Shape=[batch_size]. Sentence lengths
        """
        return self.vocab_layer(inputs)

    @tf.function(input_signature=[])
    def vocab_size(self):
        """Returns the vocabulary size of the vocab paired with the embedding

        :return int/Tensor(dtype=int)
        """
        return self._vocab_size

    @tf.function(input_signature=[])
    def num_units(self):
        """Returns the number of units (embedding size)

        :return int/tf.Tensor(dtype=int)
        """
        return self._num_units

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.dtypes.int32)])
    def embedding_lookup(self, inputs):
        """Returns the embedding of the inputs

        :param inputs Tensor(dtype=int) Shape=[batch_size, sentence_len]
        :return Tensor(dtype=float) Shape[batch_size, sentence_len, num_units]
        """
        return tf.nn.embedding_lookup(params=self.embedding, ids=inputs)


SENTENCES = tf.constant(['hello sent1', 'build sent2'])
NUM_CLS = tf.constant(0, dtype=tf.dtypes.int32)
NUM_SEP = tf.constant(0, dtype=tf.dtypes.int32)
MIN_LEN = tf.constant(0, dtype=tf.dtypes.int32)
MAX_LEN = tf.constant(5, dtype=tf.dtypes.int32)
INPUTS = get_sorted_dict({InternalFtrType.SENTENCES: SENTENCES,
                          InternalFtrType.NUM_CLS: NUM_CLS,
                          InternalFtrType.NUM_SEP: NUM_SEP,
                          InternalFtrType.MIN_LEN: MIN_LEN,
                          InternalFtrType.MAX_LEN: MAX_LEN})
TOKENIZED_RESULTS = tf.constant([[1, 2], [0, 1]])


def check_embedding_layer_api(layer: embedding_layer.EmbeddingLayerBase):
    """Checks whether the embedding layer has APIs required by DeText

    Layer that does not pass this check is definitely DeText incompatible, but passing this check does not fully guarantee DeText compatibility
    """
    layer.num_units()
    layer.vocab_size()
    layer.tokenize_to_indices(INPUTS)
    layer.embedding_lookup(TOKENIZED_RESULTS)
    layer(INPUTS)


def build_embedding_layer(vocab_hub_url, embedding_file, trainable, num_units):
    return ExampleEmbeddingLayer(vocab_hub_url, embedding_file, trainable, num_units)


def main(argv):
    argument = Args.__from_argv__(argv[1:], error_on_unknown=True)
    argument: Args

    logging.info("Building embedding layer")
    layer = build_embedding_layer(argument.vocab_hub_url, argument.embedding_file, argument.trainable, argument.num_units)

    logging.info("Checking embedding layer api")
    check_embedding_layer_api(layer)

    tf.saved_model.save(layer, argument.output_file)
    logging.info(f"Layer saved to {argument.output_file}")


if __name__ == '__main__':
    main(sys.argv)
