from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_hub as hub
from absl import logging

from smart_compose.utils.layer_utils import get_sorted_dict, get_tf_function_names
from smart_compose.utils.parsing_utils import InternalFtrType
from smart_compose.utils.vocab_utils import read_tf_vocab, read_vocab, read_tf_vocab_inverse

INPUT_SIGNATURE = get_sorted_dict({InternalFtrType.SENTENCES: tf.TensorSpec(shape=[None], dtype=tf.dtypes.string, name=InternalFtrType.SENTENCES),
                                   InternalFtrType.NUM_CLS: tf.TensorSpec(shape=[], dtype=tf.dtypes.int32, name=InternalFtrType.NUM_CLS),
                                   InternalFtrType.NUM_SEP: tf.TensorSpec(shape=[], dtype=tf.dtypes.int32, name=InternalFtrType.NUM_SEP),
                                   InternalFtrType.MIN_LEN: tf.TensorSpec(shape=[], dtype=tf.dtypes.int32, name=InternalFtrType.MIN_LEN),
                                   InternalFtrType.MAX_LEN: tf.TensorSpec(shape=[], dtype=tf.dtypes.int32, name=InternalFtrType.MAX_LEN)})


class VocabLayerBase(ABC, tf.keras.layers.Layer):
    """Vocabulary base layer that defines the interface necessary for vocabulary layer loaded from tf hub"""

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

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.dtypes.int32)])
    @abstractmethod
    def convert_ids_to_texts(self, ids):
        """Converts the indices to tokens and concatenate them into sentences

        :param ids: tf.Tensor(dtype=int) Shape=[batch_size, seq_len]
        :return string/tf.Tensor(dtype=string) Shape=[batch_size]
        """
        pass

    @tf.function(input_signature=[])
    @abstractmethod
    def sep_id(self):
        """Returns the index of the sep token

        :return int/tf.Tensor(dtype=int)
        """
        pass

    @abstractmethod
    def cls_id(self):
        """Returns the index of the cls token

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

    def adjust_len(self, inputs, min_len, max_len):
        """Adjusts the length of the inputs

        If length < min_len, padding token will be added. If length > max_len, it will be trimmed to max_len
        :param inputs Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        inputs = inputs[:, :max_len]

        padding_len = tf.maximum(0, min_len - tf.shape(inputs)[-1])
        inputs = tf.pad(inputs, paddings=[[0, 0], [0, padding_len]], constant_values=self.pad_id())
        return inputs

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
        inputs = inputs[:, tf.maximum(0, tf.shape(inputs)[1] - max_len):]

        padding_len = tf.maximum(0, min_len - tf.shape(inputs)[-1])
        inputs = tf.pad(inputs, paddings=[[0, 0], [0, padding_len]], constant_values=self.pad_id())
        return inputs

    @tf.function(input_signature=[tf.SparseTensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.int32)])
    def add_cls_sep(self, inputs, num_cls, num_sep):
        """Prepends cls tokens and appends sep tokens to the inputs

        :param inputs Token indices from vocab lookup. Sparse tensor of type int. Shape=[batch_size, sentence_len]
        :param num_cls Number of CLS to add to the start of the sentence. Scalar of type int
        :param num_sep Number of SEP to add to the start of the sentence. Scalar of type int
        """
        inputs = tf.RaggedTensor.from_sparse(inputs)
        batch_size = inputs.bounding_shape()[0]
        cls_token_shape = [batch_size, num_cls]
        sep_token_shape = [batch_size, num_sep]

        cls_tokens = tf.fill(cls_token_shape, self.cls_id())
        sep_tokens = tf.fill(sep_token_shape, self.sep_id())
        inputs = tf.concat([cls_tokens, inputs, sep_tokens], axis=1)
        inputs = self._to_dense(inputs)
        return inputs

    def _to_dense(self, inputs):
        """Converts given inputs to dense tensor"""
        if isinstance(inputs, tf.sparse.SparseTensor):
            inputs = tf.sparse.to_dense(inputs, default_value=self.pad_id())
        if isinstance(inputs, tf.RaggedTensor):
            inputs = inputs.to_tensor(default_value=self.pad_id())
        return inputs

    @tf.function(input_signature=[INPUT_SIGNATURE])
    def call(self, inputs):
        """Returns the embedding of given inputs

        :param inputs A dictionary containing
            sentences: Tensor(dtype=string) Shape=[batch_size]
            min_len: Tensor(dtype=int) Shape=[]
            max_len: Tensor(dtype=int) Shape=[]
            num_cls: Tensor(dtype=int) Shape=[]
            num_sep: Tensor(dtype=int) Shape=[]
        :return A dictionary containing
            TOKENIZED_IDS: Tensor(dtype=int) Shape=[batch_size, sentence_len]
            LENGTH: Tensor(dtype=int) Shape=[batch_size]
        """
        sentences = inputs[InternalFtrType.SENTENCES]
        num_cls = inputs[InternalFtrType.NUM_CLS]
        num_sep = inputs[InternalFtrType.NUM_SEP]

        min_len = inputs[InternalFtrType.MIN_LEN]
        max_len = inputs[InternalFtrType.MAX_LEN]

        tokenized = self.tokenize(sentences)
        indices = self.vocab_lookup(tokenized)
        padded = self.add_cls_sep(indices, num_cls, num_sep)

        trimmed_sequence = self.adjust_len(padded, min_len, max_len)  # [batch_size, sent_len]
        length = tf.reduce_sum(input_tensor=tf.cast(tf.not_equal(trimmed_sequence, self.pad_id()), dtype=tf.int32), axis=-1)
        return {InternalFtrType.TOKENIZED_IDS: trimmed_sequence, InternalFtrType.LENGTH: length}


METHODS = set(get_tf_function_names(VocabLayerBase)).union({'__call__'}).difference({'call'})


def create_vocab_layer(vocab_layer_param, vocab_hub_url):
    """Returns vocabulary layer

    If vocabulary hub url is given, loads the vocabulary layer. Otherwise, creates an vocabulary layer from given vocab layer param
    """
    if vocab_hub_url:
        logging.info(f'Loading vocab layer from {vocab_hub_url}')
        # Load saved vocab layer from given url. Loaded layer must conform to APIs defined in VocabLayerBase
        vocab_layer = hub.KerasLayer(vocab_hub_url)
        vocab_obj = vocab_layer.resolved_object
        # Add methods from resolved object to the keras layer
        for method_name in METHODS:
            setattr(vocab_layer, method_name, getattr(vocab_obj, method_name))
        return vocab_layer
    return VocabLayerFromPath(**vocab_layer_param)


class VocabLayerFromPath(VocabLayerBase):
    """Vocabulary layer

    Input text will be tokenized and convert to indices by this layer. Whitespace split is used as tokenization
    """

    def __init__(self, CLS, SEP, PAD, UNK, vocab_file):
        """ Initializes the vocabulary layer

        :param CLS Token that represents the start of a sentence
        :param SEP Token that represents the end of a segment
        :param PAD Token that represents padding
        :param UNK Token that represents unknown tokens
        :param vocab_file Path to the vocabulary file
        """
        super().__init__()
        self._vocab_table_initializer, self.vocab_table = read_tf_vocab(vocab_file, UNK)
        self._inverse_vocab_table_initializer, self.inverse_vocab_table = read_tf_vocab_inverse(vocab_file, UNK)

        self._CLS = CLS
        self._SEP = SEP
        self._PAD = PAD

        self.py_vocab_table = read_vocab(vocab_file)
        self._keys = tf.constant(list(self.py_vocab_table.keys()), dtype=tf.string)
        self._values = tf.constant(list(self.py_vocab_table.values()), dtype=tf.int32)
        self._pad_id = self.py_vocab_table[PAD]
        self._cls_id = self.py_vocab_table[CLS] if CLS else -1
        self._sep_id = self.py_vocab_table[SEP] if SEP else -1
        self._vocab_size = len(self.py_vocab_table)

    @tf.function(input_signature=[])
    def keys(self):
        return self._keys

    @tf.function(input_signature=[])
    def values(self):
        return self._values

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.dtypes.int32)])
    def convert_ids_to_texts(self, ids):
        """Converts the indices to tokens and concatenates them into sentences

        :param ids: tf.Tensor(dtype=int) Shape=[batch_size, seq_len]
        :return string/tf.Tensor(dtype=string) Shape=[batch_size]
        """
        tokens = self.inverse_vocab_table.lookup(ids)
        return tf.strings.reduce_join(tokens, axis=-1, separator=' ')

    @tf.function(input_signature=[])
    def pad_id(self):
        """Returns the index of the padding token

        :return int/tf.Tensor(dtype=int)
        """
        return self._pad_id

    @tf.function(input_signature=[])
    def vocab_size(self):
        """Returns the vocabulary size

        :return int/tf.Tensor(dtype=int)
        """
        return self._vocab_size

    def cls_id(self):
        """Returns the index of CLS token

        :return int/tf.Tensor(dtype=int)
        """
        return self._cls_id

    @tf.function(input_signature=[])
    def sep_id(self):
        """Returns the index of SEP token

        :return int/tf.Tensor(dtype=int)
        """
        return self._sep_id

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.string)])
    def vocab_lookup(self, inputs):
        """Converts given input tokens into indices

        :param inputs Tensor(dtype=string) Shape=[batch_size, sentence_len]. This is the output from _tokenize() method
        :return Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        return self.vocab_table.lookup(inputs)

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
    def tokenize(self, inputs):
        """Converts given input into tokens

        :param inputs Tensor(dtype=string) Shape=[batch_size]
        :return Tensor(dtype=string) Shape=[batch_size, sentence_len]. Output should be either dense or sparse. Ragged tensor is not supported for now
        """
        return tf.strings.split(inputs).to_sparse()
