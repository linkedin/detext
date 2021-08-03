import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.python.ops import lookup_ops
import re
import string

SENTENCEPIECE = "sentencepiece"
SPACE = "space"
WORDPIECE = "wordpiece"


class BertPreprocessLayer(tf.keras.layers.Layer):
    """Preprocess layer for Bert

    Input text will be firstly tokenized and convert to indices.
    Then CLS id and SEP id will be added indices.
    Finally, padding and truncatting will be applied
    """
    def __init__(self, bert_layer, max_len, min_len=1, CLS='[CLS]', SEP='[SEP]', PAD='[PAD]', UNK='[UNK]'):
        """ Initializes the layer

        :param CLS Token that represents the start of a sentence
        :param SEP Token that represents the end of a segment
        :param PAD Token that represents padding
        :param UNK Token that represents unknown tokens
        :param bert_layer Keras layer that loaded from pretrained BERT
        """
        super().__init__()
        self._CLS = CLS
        self._SEP = SEP
        self._PAD = PAD
        self._min_len = min_len
        self._max_len = max_len

        resolved_object = bert_layer.resolved_object
        self.do_lower_case = resolved_object.do_lower_case.numpy()
        if hasattr(resolved_object, "tokenizer_type"):
            tokenizer_type_file = resolved_object.tokenizer_type.asset_path.numpy().decode("utf-8")
            with tf.io.gfile.GFile(tokenizer_type_file, 'r') as f_handler:
                self._tokenizer_type = f_handler.read().strip()
            tokenizer_file = resolved_object.tokenizer_file.asset_path.numpy().decode("utf-8")
            if self._tokenizer_type == SENTENCEPIECE:
                with tf.io.gfile.GFile(tokenizer_file, 'rb') as f_handler:
                    sp_model = f_handler.read()
                self._tokenizer = tf_text.SentencepieceTokenizer(model=sp_model, out_type=tf.int32)
                self.vocab_table = create_tf_vocab_from_sp_tokenizer(self._tokenizer, num_oov_buckets=1)
            else:
                assert(self._tokenizer_type == SPACE)
                _, self.vocab_table = read_tf_vocab(tokenizer_file, UNK)
        else:
            vocab_file = resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
            _, self.vocab_table = create_tf_vocab_from_wp_tokenizer(vocab_file, num_oov_buckets=1)
            self._tokenizer = tf_text.BertTokenizer(self.vocab_table, token_out_type=tf.int64, lower_case=self.do_lower_case, unknown_token=UNK)
            self._tokenizer_type = WORDPIECE

        self._pad_id = self.vocab_table.lookup(tf.constant(PAD)) if PAD else -1
        self._cls_id = self.vocab_table.lookup(tf.constant(CLS)) if CLS else -1
        self._sep_id = self.vocab_table.lookup(tf.constant(SEP)) if SEP else -1

        if self._tokenizer_type == SENTENCEPIECE:
            self._pad_id = tf.cast(self._pad_id, tf.int32)
            self._cls_id = tf.cast(self._cls_id, tf.int32)
            self._sep_id = tf.cast(self._sep_id, tf.int32)

    @tf.function(input_signature=[])
    def pad_id(self):
        """Returns the index of the padding token

        :return int/tf.Tensor(dtype=int)
        """
        return self._pad_id

    def cls_id(self):
        """Returns the index of CLS token

        :return int/tf.Tensor(dtype=int)
        """
        return self._cls_id

    def sep_id(self):
        """Returns the index of SEP token

        :return int/tf.Tensor(dtype=int)
        """
        return self._sep_id

    def _vocab_lookup(self, inputs):
        """Converts given input tokens into indices

        :param inputs Tensor(dtype=string) Shape=[batch_size, sentence_len]. This is the output from _tokenize() method
        :return Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        return self.vocab_table.lookup(inputs)

    def _to_dense(self, inputs):
        """Converts given inputs to dense tensor"""
        if isinstance(inputs, tf.sparse.SparseTensor):
            inputs = tf.sparse.to_dense(inputs, default_value=self.pad_id())
        if isinstance(inputs, tf.RaggedTensor):
            inputs = inputs.to_tensor(default_value=self.pad_id())
        return inputs

    def _tokenize(self, inputs):
        """Converts given input into indices

        :param inputs Tensor(dtype=string) Shape=[batch_size]
        :return Tensor(dtype=int) Shape=[batch_size, sentence_len]. Output should be Ragged tensor
        """
        if self._tokenizer_type == SPACE:
            if self.do_lower_case:
                inputs = tf.strings.lower(inputs)
            inputs = tf.strings.split(inputs).to_sparse()
            tokens = self._vocab_lookup(inputs)
        elif self._tokenizer_type == SENTENCEPIECE:
            # The sentencepiece tokenizer does not handle lowercasing
            if self.do_lower_case:
                inputs = tf.strings.lower(inputs)
            tokens = self._tokenizer.tokenize(inputs)
        elif self._tokenizer_type == WORDPIECE:
            tokens = self._tokenizer.tokenize(inputs)
            # The tf_text bert_tokenizer adds another axis which needs be sequeezed
            tokens = tokens.merge_dims(1, 2)
        else:
            raise ValueError(f'Unsupported tokenization method: {self._tokenizer_type}')
        return tokens

    def adjust_len(self, inputs, min_len, max_len):
        """Adjusts the length of the inputs

        If length < min_len, padding token will be added. If length > max_len, inputs will be trimmed to max_len
        :param inputs Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        inputs = inputs[:, :max_len]

        padding_len = tf.maximum(0, min_len - tf.shape(inputs)[-1])
        inputs = tf.pad(inputs, paddings=[[0, 0], [0, padding_len]], constant_values=self.pad_id())
        return inputs

    def add_cls_sep(self, inputs, num_cls=1, num_sep=1):
        """Prepends cls tokens and appends sep tokens to the inputs"""
        if self._tokenizer_type == SPACE:
            inputs = tf.RaggedTensor.from_sparse(inputs)
        batch_size = inputs.bounding_shape()[0]
        cls_token_shape = [batch_size, num_cls]
        sep_token_shape = [batch_size, num_sep]

        cls_tokens = tf.fill(cls_token_shape, self.cls_id())
        sep_tokens = tf.fill(sep_token_shape, self.sep_id())
        inputs = tf.concat([cls_tokens, inputs, sep_tokens], axis=1)

        # Convert tensor to dense
        if self._tokenizer_type == SPACE:
            inputs = self._to_dense(inputs)
        else:
            inputs = tf_text.keras.layers.ToDense(pad_value=self.pad_id())(inputs)

        return inputs

    @tf.function()
    def call(self, inputs):
        """Returns the indices of given inputs

        :param
            inputs: Tensor(dtype=string) Shape=[batch_size]
        :return
            result: Tensor(dtype=int) Shape=[batch_size, sentence_len]
        """
        indices = self._tokenize(inputs)
        padded = self.add_cls_sep(indices)
        result = self.adjust_len(padded, self._min_len, self._max_len)  # [batch_size, sent_len]

        input_ids = tf.cast(result, tf.int32)

        return input_ids


def create_tf_vocab_from_sp_tokenizer(sp_tokenizer, num_oov_buckets):
    """Create a lookup table for a vocabulary from sp tokenizer"""
    vocab_size = sp_tokenizer.vocab_size().numpy()
    vocab_values = tf.range(vocab_size, dtype=tf.int64)
    vocab = []
    for i in range(vocab_size):
        vocab.append(sp_tokenizer.id_to_string(i))
    init = tf.lookup.KeyValueTensorInitializer(keys=vocab, values=vocab_values, key_dtype=tf.string, value_dtype=tf.int64)
    vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets, lookup_key_dtype=tf.string)
    return vocab_table


def create_tf_vocab_from_wp_tokenizer(input_file, num_oov_buckets):
    """Read vocabulary and return a tf hashtable that can be used for tensorflow text bert tokenizer"""
    if input_file is None:
        return None

    keys, values = [], []
    fin = tf.io.gfile.GFile(input_file, 'r')
    for line in fin:
        word = split(strip(line))[0]
        keys.append(word)
        values.append(len(values))
    fin.close()
    # The output type has to be tf.int64 due to the constraint of tf.lookup.StaticVocabularyTable
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values, dtype=tf.int64))
    vocab_table = tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets=num_oov_buckets, lookup_key_dtype=tf.string)
    return initializer, vocab_table


def read_tf_vocab(input_file, UNK):
    """Read vocabulary and return a tf hashtable"""
    if input_file is None:
        return None

    keys, values = [], []
    fin = tf.io.gfile.GFile(input_file, 'r')
    for line in fin:
        word = split(strip(line))[0]
        keys.append(word)
        values.append(len(values))
    fin.close()
    UNK_ID = keys.index(UNK)

    initializer = lookup_ops.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    vocab_table = lookup_ops.HashTable(initializer, UNK_ID)
    return initializer, vocab_table


def strip(s):
    """Strips ascii whitespace characters off string s."""
    return s.strip(string.whitespace)


def split(s):
    """Split string s by whitespace characters."""
    whitespace_lst = [re.escape(ws) for ws in string.whitespace]
    pattern = re.compile('|'.join(whitespace_lst))
    return pattern.split(s)
