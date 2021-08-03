from abc import abstractmethod, ABC

import tensorflow as tf
import tensorflow_hub as hub
from absl import logging

from detext.utils.parsing_utils import InternalFtrType
from detext.layers import vocab_layer
from detext.layers.vocab_layer import create_vocab_layer
from detext.utils.layer_utils import init_word_embedding


class EmbeddingLayerBase(ABC, tf.keras.layers.Layer):
    """Embedding base layer that defines the interface necessary for embedding layer loaded from tf hub

    All abstract methods need to be decorated with @tf.function
    """

    @tf.function
    @abstractmethod
    def tokenize_to_indices(self, inputs):
        """Tokenize given inputs and convert to indices

        Example: tokenize_to_indices(['hello world', 'sentence 1 token']) -> {TOKENIZED_IDS: [[20, 10, pad_id], [4, 5, 6]], LENGTH: [2, 3]}

        :param inputs tf.Tensor(dtype=string) Shape=[batch_size]
        :return A dictionary containing the following key values:
          TOKENIZED_IDS: tf.Tensor(dtype=int) Shape=[batch_size, sentence_len]. Tokenization and lookup result
          LENGTH: tf.Tensor(dtype=int) Shape=[batch_size]. Sentence lengths
        """
        pass

    @tf.function
    @abstractmethod
    def vocab_size(self):
        """Returns the vocabulary size of the vocab paired with the embedding

        :return int/Tensor(dtype=int)
        """
        pass

    @tf.function
    @abstractmethod
    def embedding_lookup(self, inputs):
        """Returns the embedding of the inputs

        :param inputs Tensor(dtype=int) Shape=[batch_size, sentence_len]
        :return Tensor(dtype=float) Shape[batch_size, sentence_len, num_units]
        """
        pass

    @tf.function
    @abstractmethod
    def num_units(self):
        """Returns the number of units (embedding size)

        :return int/tf.Tensor(dtype=int)
        """
        pass

    @tf.function(input_signature=[vocab_layer.INPUT_SIGNATURE])
    def call(self, inputs):
        """Returns the embedding of given inputs

        :param inputs A dictionary containing
            SENTENCES: Tensor(dtype=string) Shape=[batch_size]
            MIN_LEN: Tensor(dtype=int) Shape=[]
            MAX_LEN: Tensor(dtype=int) Shape=[]
            NUM_CLS: Tensor(dtype=int) Shape=[]
            NUM_SEP: Tensor(dtype=int) Shape=[]
        :return A dictionary containing
            TOKENIZED_IDS: Tensor(dtype=int) Shape=[batch_size, sentence_len]
            EMBEDDED: Tensor(dtype=float) Shape=[batch_size, sentence_len, num_units]
            LENGTH: Tensor(dtype=int) Shape=[batch_size]
        """
        inputs = self.tokenize_to_indices(inputs)  # {TOKENIZED_IDS: ..., LENGTH: ...}
        inputs[InternalFtrType.EMBEDDED] = self.embedding_lookup(inputs[InternalFtrType.TOKENIZED_IDS])  # [batch_size, sent_len, num_units]
        return inputs


def create_embedding_layer(embedding_layer_param, embedding_hub_url):
    """Returns an embedding layer

    If embedding hub url is given, loads the embedding layer. Otherwise, creates an embedding layer from given embedding_layer_param
    """
    if embedding_hub_url:
        logging.info(f'Loading pretrained embedding layer from {embedding_hub_url}')
        # Load saved embedding layer from given url. Loaded layer must conform to APIs defined in EmbeddingLayerBase
        embedding_layer = hub.KerasLayer(embedding_hub_url, trainable=embedding_layer_param['we_trainable'])
        embedding_obj = embedding_layer.resolved_object
        # Add methods from resolved object to the keras layer
        for method_name in ['__call__', 'tokenize_to_indices', 'num_units', 'vocab_size', 'embedding_lookup']:
            setattr(embedding_layer, method_name, getattr(embedding_obj, method_name))
        return embedding_layer
    return EmbeddingLayer(**embedding_layer_param)


class EmbeddingLayer(EmbeddingLayerBase):
    """Embedding layer"""

    def __init__(self, vocab_layer_param, vocab_hub_url, we_file, we_trainable, num_units, name_prefix='w'):
        """ Initializes the embedding layer

        :param vocab_layer_param Parameters related to vocabulary layer initialization. If vocab_hub_url is empty/None, a new vocab layer will be constructed
          using this param
        :param vocab_hub_url Url to saved vocabulary layer. If empty string or None, no vocab layer will be loaded
        :param we_file Path to pretrained word embedding
        :param we_trainable Whether word embedding is trainable
        :param num_units Dimension of embedding
        :param name_prefix Prefix of embedding variables
        """
        super().__init__()
        self.vocab_layer = create_vocab_layer(vocab_layer_param, vocab_hub_url=vocab_hub_url)
        self._num_units = num_units
        self._vocab_size = self.vocab_layer.vocab_size()

        self.embedding = init_word_embedding(self._vocab_size, num_units, we_trainable, we_file, name_prefix)

    @tf.function
    def tokenize_to_indices(self, inputs):
        """Tokenize given inputs and convert to indices

        Example: tokenize_to_indices(['hello world', 'sentence 1 token']) -> {TOKENIZED_IDS: [[20, 10, pad_id], [4, 5, 6]], LENGTH: [2, 3]}

        :param inputs tf.Tensor(dtype=string) Shape=[batch_size]
        :return A dictionary containing the following key values:
          TOKENIZED_IDS: tf.Tensor(dtype=int) Shape=[batch_size, sentence_len]. Tokenization and lookup result
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
