from abc import abstractmethod, ABC

import tensorflow as tf
import tensorflow_hub as hub
from absl import logging

from smart_compose.layers import vocab_layer
from smart_compose.layers.vocab_layer import create_vocab_layer
from smart_compose.utils.layer_utils import init_word_embedding, get_tf_function_names
from smart_compose.utils.parsing_utils import InternalFtrType


class EmbeddingLayerBase(ABC, tf.keras.layers.Layer):
    """Embedding base layer that defines the interface necessary for embedding layer loaded from tf hub

    All abstract methods need to be decorated with @tf.function
    """

    @tf.function(input_signature=[])
    @abstractmethod
    def vocab_size(self):
        """Returns the vocabulary size

        :return int/tf.Tensor(dtype=int)
        """
        pass

    @tf.function(input_signature=[])
    @abstractmethod
    def pad_id(self):
        """Returns the index of the padding token

        :return int/tf.Tensor(dtype=int)
        """
        pass

    @tf.function
    @abstractmethod
    def sep_id(self):
        """Returns the index of the sep token

        :return int/tf.Tensor(dtype=int)
        """
        pass

    @tf.function(input_signature=[tf.SparseTensorSpec([None, None], dtype=tf.string)])
    @abstractmethod
    def vocab_lookup(self, inputs):
        """Converts given input tokens into indices

        :param inputs Tensor(dtype=string) Shape=[batch_size, sentence_len]. This is the output from _tokenize() method
        :return Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        pass

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
    @abstractmethod
    def tokenize(self, inputs):
        """Converts given input into tokens

        :param inputs Tensor(dtype=string) Shape=[batch_size]
        :return Tensor(dtype=string) Shape=[batch_size, sentence_len]
        """
        pass

    @tf.function(input_signature=[])
    @abstractmethod
    def keys(self):
        """Returns the keys (tokens) in the vocab table

        :return Tensor(dtype=string) Shape=[num_keys]
        """
        pass

    @tf.function(input_signature=[])
    @abstractmethod
    def values(self):
        """Returns the values (token index) in the vocab table

        :return Tensor(dtype=string) Shape=[num_values]
        """
        pass

    @abstractmethod
    @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32)])
    def adjust_len_right(self, inputs, min_len, max_len):
        """Adjusts the length of the inputs. Different than adjust_len(), this function  only keeps the last max_len tokens if length > max_len. This function
            is helpful for online serving of writing assistance where the most relevance words are the words that the users are currently typing and thus
            it's more helpful to keep the last max_len words

        If length < min_len, padding token will be added. If length > max_len, only the last max_len will be kept
        :param inputs Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        pass

    @tf.function(input_signature=[tf.SparseTensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32)])
    def add_cls_sep(self, inputs, num_cls, num_sep):
        """Prepends cls tokens and appends sep tokens to the inputs

        :param inputs Token indices from vocab lookup. Sparse tensor of type int. Shape=[batch_size, sentence_len]
        :param num_cls Number of CLS to add to the start of the sentence. Scalar of type int
        :param num_sep Number of SEP to add to the start of the sentence. Scalar of type int
        """
        pass

    @tf.function
    @abstractmethod
    def tokenize_to_indices(self, inputs):
        """Tokenize given inputs and convert to indices

        Example: tokenize_to_indices(['hello world', 'sentence 1 token']) -> {RESULT: [[20, 10, pad_id], [4, 5, 6]], LENGTH: [2, 3]}

        :param inputs tf.Tensor(dtype=string) Shape=[batch_size]
        :return A dictionary containing the following key values:
          TOKENIZED: tf.Tensor(dtype=int) Shape=[batch_size, sentence_len]. Tokenization and lookup result
          LENGTH: tf.Tensor(dtype=int) Shape=[batch_size]. Sentence lengths
        """
        pass

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.dtypes.int32)])
    @abstractmethod
    def convert_ids_to_texts(self, ids):
        """Converts the indices to tokens and concatenate them into sentences

        :param ids: tf.Tensor(dtype=int) Shape=[batch_size, seq_len]
        :return string/tf.Tensor(dtype=string) Shape=[batch_size]
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
            sentences: Tensor(dtype=string) Shape=[batch_size]
            min_len: Tensor(dtype=int) Shape=[]
            max_len: Tensor(dtype=int) Shape=[]
            num_cls: Tensor(dtype=int) Shape=[]
            num_sep: Tensor(dtype=int) Shape=[]
        :return A dictionary containing
            embedded: Tensor(dtype=float) Shape=[batch_size, sentence_len, num_units]
            tokenized: Tensor(dtype=int) Shape=[batch_size, sentence_len]
            length: Tensor(dtype=int) Shape=[batch_size]
        """
        inputs = self.tokenize_to_indices(inputs)  # {'tokenized': ..., 'length': ...}
        inputs[InternalFtrType.EMBEDDED] = self.embedding_lookup(inputs[InternalFtrType.TOKENIZED_IDS])  # [batch_size, sent_len, num_units]
        return inputs


METHODS = vocab_layer.METHODS.union(get_tf_function_names(EmbeddingLayerBase)).union({'__call__'}).difference({'call'})


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
        for method_name in METHODS:
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
        self._sep_id = self.vocab_layer.sep_id()

        self.embedding = init_word_embedding(self._vocab_size, num_units, we_trainable, we_file, name_prefix)

    @tf.function(input_signature=[])
    def sep_id(self):
        return self._sep_id

    @tf.function
    def tokenize_to_indices(self, inputs):
        """Tokenize given inputs and convert to indices

        Example: tokenize_to_indices(['hello world', 'sentence 1 token']) -> {RESULT: [[20, 10, pad_id], [4, 5, 6]], LENGTH: [2, 3]}

        :param inputs tf.Tensor(dtype=string) Shape=[batch_size]
        :return A dictionary containing the following key values:
          TOKENIZED: tf.Tensor(dtype=int) Shape=[batch_size, sentence_len]. Tokenization and lookup result
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

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.dtypes.int32)])
    def convert_ids_to_texts(self, ids):
        """Converts the indices to tokens and concatenate them into sentences

        :param ids: tf.Tensor(dtype=int) Shape=[batch_size, seq_len]
        :return string/tf.Tensor(dtype=string) Shape=[batch_size]
        """
        return self.vocab_layer.convert_ids_to_texts(ids)

    @tf.function(input_signature=[])
    def pad_id(self):
        """Returns the index of the padding token

        :return int/tf.Tensor(dtype=int)
        """
        return self.vocab_layer.pad_id()

    @tf.function(input_signature=[tf.SparseTensorSpec([None, None], dtype=tf.string)])
    def vocab_lookup(self, inputs):
        """Converts given input tokens into indices

        :param inputs Tensor(dtype=string) Shape=[batch_size, sentence_len]. This is the output from _tokenize() method
        :return Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        return self.vocab_layer.vocab_lookup(inputs)

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
    def tokenize(self, inputs):
        """Converts given input into tokens

        :param inputs Tensor(dtype=string) Shape=[batch_size]
        :return Tensor(dtype=string) Shape=[batch_size, sentence_len]
        """
        return self.vocab_layer.tokenize(inputs)

    @tf.function(input_signature=[])
    def keys(self):
        """Returns the keys (tokens) in the vocab table

        :return Tensor(dtype=string) Shape=[num_keys]
        """
        return self.vocab_layer.keys()

    @tf.function(input_signature=[])
    def values(self):
        """Returns the values (token index) in the vocab table

        :return Tensor(dtype=string) Shape=[num_values]
        """
        return self.vocab_layer.values()

    @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32)])
    def adjust_len_right(self, inputs, min_len, max_len):
        """Adjusts the length of the inputs. Different than adjust_len(), this function  only keeps the last max_len tokens if length > max_len. This function
            is helpful for online serving of writing assistance where the most relevance words are the words that the users are currently typing and thus
            it's more helpful to keep the last max_len words

        If length < min_len, padding token will be added. If length > max_len, only the last max_len will be kept
        :param inputs Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        return self.vocab_layer.adjust_len_right(inputs, min_len, max_len)

    @tf.function(input_signature=[tf.SparseTensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32)])
    def add_cls_sep(self, inputs, num_cls, num_sep):
        """Prepends cls tokens and appends sep tokens to the inputs

        :param inputs Token indices from vocab lookup. Sparse tensor of type int. Shape=[batch_size, sentence_len]
        :param num_cls Number of CLS to add to the start of the sentence. Scalar of type int
        :param num_sep Number of SEP to add to the start of the sentence. Scalar of type int
        """
        return self.vocab_layer.add_cls_sep(inputs, num_cls, num_sep)
