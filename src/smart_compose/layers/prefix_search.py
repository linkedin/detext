from collections import defaultdict

import tensorflow as tf

from smart_compose.utils.parsing_utils import InternalFtrType


class KeyValueArrayDict(tf.Module):
    """A dictionary holding mapping of key -> sparse values

    In prefix lookup, we need to return tokens starting with user input. A naive approach is to build a mapping from all possible prefixes to
        the tokens. Since the numbers of tokens are different regarding different user inputs, the values of the mapping are sparse. The native
        TF hashtable (e.g. StaticHashTable) does not support value as sparse tensor.

    To suffice the needs on sparse value lookup, we implement a custom lookup table supporting keys of arbitrary TF scalar type and sparse values of
        arbitrary TF data type. Due to the limitation of TF sparse tensor operation, this data structure only supports scalar lookup (key can only be
        a scalar)
    """
    dtype_to_default_value = {
        tf.string: "",
        tf.int32: -1,
        tf.float32: 1.0
    }

    def __init__(self, keys, values):
        """ Initializes the dictionary

        :param keys: Array of keys. E.g. ['k0', 'k1', ...]
        :param values: Array of value arrays. E.g. [['k0_v0', 'k0_v1'], ['k1_v0'], ...]
        """
        assert len(keys) == len(set(keys)), "Keys must be unique"
        assert len(keys) == len(values), "Keys and values must have the same shape"
        super().__init__()
        values = [sorted(array) for array in values]
        self.key_dtype = self._get_dtype(keys[0])
        self.key_default_value = self.dtype_to_default_value[self.key_dtype]
        self.value_dtype = self._get_dtype(values[0][0])

        # Sort values and build a mapping of value -> value index
        all_values = set()
        for array in values:
            array = sorted(array)
            all_values.update(array)
        all_values = sorted(list(all_values))
        value_to_index = dict(list(zip(all_values, range(len(all_values)))))

        self.num_keys = len(keys)
        self.num_values = len(all_values)

        # Build a mapping of key to key index
        self.key_to_index_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys, dtype=self.key_dtype),
                                                values=tf.convert_to_tensor(
                                                    list(range(len(keys))),
                                                    dtype=tf.int32
                                                )),
            default_value=self.dtype_to_default_value[tf.int32]
        )

        # Collect the keys and values to build the sparse tensor
        sparse_tensor_indices = []
        for key, array in zip(keys, values):
            key_index = self.key_to_index_table.lookup(tf.convert_to_tensor(key)).numpy()
            for value in array:
                value_index = value_to_index[value]
                sparse_tensor_indices.append([key_index, value_index])
        sparse_tensor_indices = tf.convert_to_tensor(sparse_tensor_indices, dtype=tf.int64)

        sparse_tensor_values = []
        for array in values:
            sparse_tensor_values += array
        sparse_tensor_values = tf.convert_to_tensor(sparse_tensor_values, dtype=self.value_dtype)

        # SparseTensor: rank = 2, first axis is key index, second axis is related value indices list
        self.values_sparse_tensor = tf.SparseTensor(
            indices=sparse_tensor_indices,  # rank 2, [number of keys * number of integers in the value
            values=sparse_tensor_values,  # rank 1
            dense_shape=[len(keys), len(all_values)]
        )

    def _get_dtype(self, val):
        """Returns the dtype of value"""
        if isinstance(val, (str, bytes)):
            return tf.string
        if isinstance(val, int):
            return tf.int32
        if isinstance(val, float):
            return tf.float32
        raise ValueError(f"Unsupported type {type(val)}")

    @tf.function
    def lookup(self, key):
        """Returns the values related to the key

        :param key: A SCALAR of the same type as the type of keys used to build the dictionary
        """
        index = self.key_to_index_table.lookup(key)
        exist_key = (index != self.dtype_to_default_value[tf.int32])
        return {
            InternalFtrType.COMPLETION_INDICES: tf.sparse.reshape(
                tf.sparse.slice(self.values_sparse_tensor, [index, 0], [1, self.num_values]),
                shape=[self.num_values]
            ),
            InternalFtrType.EXIST_KEY: exist_key
        }


class PrefixLookupTable(tf.Module):
    """Prefix lookup table that returns the indices of tokens starting with the given prefix """

    def __init__(self, vocab):
        """Initializes the prefix lookup table

        :param vocab: Mapping of token -> index
        """
        super().__init__()
        self.prefix_table_py = self.build_prefix_dict(vocab)
        self.prefix_table_tf = KeyValueArrayDict(list(self.prefix_table_py.keys()), list(self.prefix_table_py.values()))

    def build_prefix_dict(self, vocab):
        """Returns the dictionary that maps prefix to list of words starting with the prefix """
        prefix_to_completions = defaultdict(list)
        for w, idx in vocab.items():
            for i in range(len(w) + 1):
                prefix_to_completions[w[:i]].append(idx)

        return prefix_to_completions

    def get_vocab_mask(self, word_indices):
        """Returns a vocabulary mask where only values at word indices are 1s

        :param word_indices: tf.SparseTensor(dtype=int32, shape=[..., vocab_size]
        :return vocab_mask: tf.Tensor(dtype=bool, shape=[..., vocab_size]
        """
        return tf.sparse.to_indicator(word_indices, vocab_size=tf.shape(word_indices)[-1])

    def __call__(self, inputs):
        """ Looks up all tokens starting with given prefix

        :param inputs: tf.Tensor(dtype=string, shape=[])
        :return: A dictionary containing {
            InternalFtrType.EXIST_PREFIX: lookup_results[InternalFtrType.EXIST_PREFIX],
            InternalFtrType.COMPLETION_VOCAB_MASK: mask
        }
        """
        lookup_results = self.prefix_table_tf.lookup(inputs)  # [1, vocab_size]
        mask = self.get_vocab_mask(lookup_results[InternalFtrType.COMPLETION_INDICES])  # [vocab_size]
        return {
            InternalFtrType.EXIST_PREFIX: lookup_results[InternalFtrType.EXIST_KEY],
            InternalFtrType.COMPLETION_VOCAB_MASK: mask
        }


class PrefixSearcher(tf.Module):
    """Next token searcher that searches for vocabulary tokens that start with last token of the input text """

    def __init__(self, vocab_layer, min_len, max_len, num_cls, num_sep):
        """ Initializes the searcher

        :param vocab_layer: the vocabulary layer that provides the keys, values, and performs tokenization
        :param min_len: int. Min length of the input sentence. If sentence with length smaller than this value will be padded to min_len
        :param max_len: int. Max length of the input sentence. If sentence with length larger than this value will be trimmed to max_len
        :param num_cls: int. Number of CLS token to add to the sentence
        :param num_sep: int. Number of SEP token to add to the sentence
        """
        super().__init__()
        self.vocab_layer = vocab_layer
        self.prefix_lookup_table = PrefixLookupTable(
            self.assemble_vocab_dict(vocab_layer.keys().numpy().tolist(),
                                     vocab_layer.values().numpy().tolist())
        )
        self.vocab_size = vocab_layer.vocab_size().numpy()
        self.min_len = min_len
        self.max_len = max_len
        self.num_cls = num_cls
        self.num_sep = num_sep

    def assemble_vocab_dict(self, keys, values):
        """Assembles vocabulary dictionary from given keys and values """
        py_keys = []
        for k in keys:
            py_keys.append(k)

        py_values = []
        for v in values:
            py_values.append(v)
        return dict(zip(py_keys, py_values))

    def tokenize_and_lookup_prefix(self, sentence, min_len, max_len, num_cls, num_sep):
        """Tokenizes and look up tokens starting with the last token of the given sentence

        :param sentence: Tensor(dtype=string, shape=[]). Input text
        :param min_len: int. Min length of the input sentence. If sentence with length smaller than this value will be padded to min_len
        :param max_len: int. Max length of the input sentence. If sentence with length larger than this value will be trimmed to max_len
        :param num_cls: int. Number of CLS token to add to the sentence
        :param num_sep: int. Number of SEP token to add to the sentence
        """
        sentence = tf.expand_dims(sentence, axis=0)
        tokenized = self.vocab_layer.tokenize(sentence)  # string [1, sentence_length]
        tokenized_dense = tf.sparse.to_dense(tokenized, default_value="")

        # Treat last token as prefix and look up tokens startign with the prefix
        last_token = tokenized_dense[0, -1]
        prefix_lookup_results = self.prefix_lookup_table(last_token)

        # Convert tokens except the last one from the input sentence to indices
        prev_sequence = tokenized_dense[:, :-1]
        indices = self.vocab_layer.vocab_lookup(tf.sparse.from_dense(prev_sequence))
        padded = self.vocab_layer.add_cls_sep(indices, num_cls, num_sep)
        trimmed_sequence = self.vocab_layer.adjust_len_right(padded, min_len, max_len)  # [batch_size, sent_len]
        length = tf.reduce_sum(input_tensor=tf.cast(tf.not_equal(trimmed_sequence, self.vocab_layer.pad_id()), dtype=tf.int32), axis=-1)
        return {
            InternalFtrType.EXIST_PREFIX: prefix_lookup_results[InternalFtrType.EXIST_PREFIX],
            InternalFtrType.SEQUENCE_TO_ENCODE: tf.squeeze(trimmed_sequence, axis=0),
            InternalFtrType.LENGTH: tf.squeeze(length, axis=0),  # length of prefix to encode
            InternalFtrType.COMPLETION_VOCAB_MASK: prefix_lookup_results[InternalFtrType.COMPLETION_VOCAB_MASK]
        }

    def tokenize(self, sentence, min_len, max_len, num_cls, num_sep):
        """Tokenizes the given sentence and convert it to ids

        :param sentence: Tensor(dtype=string, shape=[]). Input text
        :param min_len: int. Min length of the input sentence. If sentence with length smaller than this value will be padded to min_len
        :param max_len: int. Max length of the input sentence. If sentence with length larger than this value will be trimmed to max_len
        :param num_cls: int. Number of CLS token to add to the sentence
        :param num_sep: int. Number of SEP token to add to the sentence
        """
        sentence = tf.expand_dims(sentence, axis=0)
        tokenized = self.vocab_layer.tokenize(sentence)  # string [1, sentence_length]
        indices = self.vocab_layer.vocab_lookup(tokenized)
        padded = self.vocab_layer.add_cls_sep(indices, num_cls, num_sep)
        trimmed_sequence = self.vocab_layer.adjust_len_right(padded, min_len, max_len)  # [batch_size, sent_len]
        length = tf.reduce_sum(input_tensor=tf.cast(tf.not_equal(trimmed_sequence, self.vocab_layer.pad_id()), dtype=tf.int32), axis=-1)
        return {
            InternalFtrType.EXIST_PREFIX: True,
            InternalFtrType.SEQUENCE_TO_ENCODE: tf.squeeze(trimmed_sequence, axis=0),
            InternalFtrType.LENGTH: tf.squeeze(length, axis=0),
            InternalFtrType.COMPLETION_VOCAB_MASK: tf.cast(tf.ones([self.vocab_size], dtype=tf.int32), dtype=tf.bool),
        }

    def __call__(self, inputs):
        """ Looks up words starting with the last token of the input text

        :param inputs: tf.Tensor(dtype=string, shape=[]) Input sentence
        :return: A dictionary containing {
            InternalFtrType.EXIST_PREFIX: Tensor(dtype=bool, shape=[]). Whether there are tokens that start with the given prefix
            InternalFtrType.SEQUENCE_TO_ENCODE: Tensor(dtype=int, shape=[sentence_len]). The token ids except the prefix to be completed
            InternalFtrType.LENGTH: Tensor(dtype=int, shape=[]). Number of non padding tokens in SEQUENCE_TO_ENCODE
            InternalFtrType.COMPLETION_VOCAB_MASK: Tensor(dtype=bool, shape=[vocab_size]). Indicator mask that shows tokens starting with the given prefix
        }
        """
        if tf.strings.length(inputs) == 0:
            return {
                InternalFtrType.EXIST_PREFIX: False,
                InternalFtrType.SEQUENCE_TO_ENCODE: tf.convert_to_tensor([], dtype=tf.int32),
                InternalFtrType.LENGTH: tf.convert_to_tensor(0, dtype=tf.int32),
                InternalFtrType.COMPLETION_VOCAB_MASK: tf.cast(tf.zeros([self.vocab_size], dtype=tf.int32), dtype=tf.bool)
            }
        if tf.strings.substr(inputs, -1, 1) == tf.convert_to_tensor(" "):
            return self.tokenize(inputs, self.min_len, self.max_len, self.num_cls, self.num_sep)
        return self.tokenize_and_lookup_prefix(inputs, self.min_len, self.max_len, self.num_cls, self.num_sep)
