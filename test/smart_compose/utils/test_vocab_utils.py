import tensorflow as tf

from smart_compose.utils import vocab_utils
from smart_compose.utils.testing.test_case import TestCase


class TestVocabUtils(TestCase):
    """Unit test for vocab_utils.py"""

    def testReadVocab(self):
        """Tests read_vocab()"""
        vocab = vocab_utils.read_vocab(self.vocab_file)
        self.assertEqual(len(vocab), 16)
        self.assertEqual(vocab[self.UNK], 0)

    def testReadTfVocab(self):
        """Tests read_tf_vocab()"""
        _, vocab = vocab_utils.read_tf_vocab(self.vocab_file, self.UNK)
        self.assertEqual(vocab.size(), 16)
        self.assertEqual(vocab.lookup(tf.constant(self.UNK)), 0)

    def testReadTfVocabInverse(self):
        """Tests read_tf_vocab_inverse()"""
        _, vocab = vocab_utils.read_tf_vocab_inverse(self.vocab_file, self.UNK)
        self.assertEqual(vocab.size(), 16)
        self.assertEqual(vocab.lookup(tf.constant(self.UNK_ID)), self.UNK)


if __name__ == '__main__':
    tf.test.main()
