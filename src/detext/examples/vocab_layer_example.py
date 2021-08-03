import string
import sys
from dataclasses import dataclass

import tensorflow as tf
from absl import logging
from detext.layers import vocab_layer
from detext.utils.layer_utils import get_sorted_dict
from detext.utils.parsing_utils import InternalFtrType
from smart_arg import arg_suite


@arg_suite
@dataclass
class Args:
    vocab_file: str  # Path of the vocabulary file which contains one token each line
    output_file: str  # Path of the output layer

    CLS: str = '[CLS]'  # Start of sentence token
    SEP: str = '[SEP]'  # End of sentence token
    PAD: str = '[PAD]'  # Padding token
    UNK: str = '[UNK]'  # Unknown token


def read_vocab(input_file: str):
    """Read vocabulary file and return a dict

    :param input_file Path to input vocab file in txt format
    """
    vocab = {}
    fin = tf.io.gfile.GFile(input_file, 'r')
    for line in fin:
        line = line.strip(string.whitespace)
        word = line.split()[0]
        vocab[word] = len(vocab)
    fin.close()
    return vocab


def read_tf_vocab(input_file: str, UNK: str):
    """Read vocabulary and return a tf hashtable

    :param input_file Path to input vocab file in txt format
    :param token for unknown words
    """
    keys, values = [], []
    fin = tf.io.gfile.GFile(input_file, 'r')
    for line in fin:
        line = line.strip(string.whitespace)
        word = line.split()[0]
        keys.append(word)
        values.append(len(values))
    fin.close()
    UNK_ID = keys.index(UNK)

    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    vocab_table = tf.lookup.StaticHashTable(initializer, UNK_ID)
    return initializer, vocab_table


class ExampleVocabLayer(vocab_layer.VocabLayerBase):
    """Example vocabulary layer accepted by DeText

    To check whether a given vocab layer conforms to the DeText API, follow the test function in detext unit tests:
      test/layers/test_vocab_layer.testVocabLayerApi()

    Input text will be tokenized and convert to indices by this layer. Whitespace split is used as tokenization
    """

    def __init__(self, CLS: str, SEP: str, PAD: str, UNK: str, vocab_file: str):
        """ Initializes the vocabulary layer

        :param CLS Token that represents the start of a sentence
        :param SEP Token that represents the end of a segment
        :param PAD Token that represents padding
        :param UNK Token that represents unknown tokens
        :param vocab_file Path to the vocabulary file
        """
        super().__init__()
        self._vocab_table_initializer, self.vocab_table = read_tf_vocab(vocab_file, UNK)

        self._CLS = CLS
        self._SEP = SEP
        self._PAD = PAD

        py_vocab_table = read_vocab(vocab_file)
        self._pad_id = py_vocab_table[PAD]
        self._cls_id = py_vocab_table[CLS] if CLS else -1
        self._sep_id = py_vocab_table[SEP] if SEP else -1
        self._vocab_size = len(py_vocab_table)

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

    def _tokenize(self, inputs):
        """Converts given input into tokens

        :param inputs Tensor(dtype=string) Shape=[batch_size]
        :return Tensor(dtype=string) Shape=[batch_size, sentence_len]. Output should be either dense or sparse. Ragged tensor is not supported for now
        """
        return tf.strings.split(inputs).to_sparse()


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


def check_vocab_layer_api(layer: vocab_layer.VocabLayerBase):
    """Checks whether the vocab layer has APIs required by DeText

    Layer that does not pass this check is definitely DeText incompatible, but passing this check does not fully guarantee DeText compatibility
    """
    layer(INPUTS)


def build_vocab_layer(CLS: str, SEP: str, PAD: str, UNK: str, vocab_file: str):
    return ExampleVocabLayer(CLS, SEP, PAD, UNK, vocab_file)


def main(argv):
    argument = Args.__from_argv__(argv[1:], error_on_unknown=True)
    argument: Args

    logging.info("Building vocab layer")
    layer = build_vocab_layer(CLS=argument.CLS, SEP=argument.SEP, PAD=argument.PAD, UNK=argument.UNK, vocab_file=argument.vocab_file)

    logging.info("Checking vocab layer api")
    check_vocab_layer_api(layer)

    tf.saved_model.save(layer, argument.output_file)
    logging.info(f"Layer saved to {argument.output_file}")


if __name__ == '__main__':
    main(sys.argv)
