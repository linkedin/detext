# -*- coding: utf-8 -*-

import gzip
import os
import six
import tensorflow as tf
from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE

from detext.utils import vocab_utils


def switch_to(mode):
    """Switches tensorflow execution mode to mode (EAGER_MODE/GRAPH_MODE)."""
    ctx = context()._eager_context
    ctx.mode = mode
    ctx.is_eager = mode == EAGER_MODE


class TestVocab(tf.test.TestCase):
    """Unit test for vocab_utils.py."""

    def testSplit(self):
        # Test case for ascii whitespace characters
        s = 'a*($*#(*$& b c'
        self.assertAllEqual(['a*($*#(*$&', 'b', 'c'], vocab_utils.split(s))

        # This string is not empty, it contains a non-ascii whitespace character. Run print(len(s)) to check
        s = u' '
        self.assertFalse(len(s) == 0)
        self.assertAllEqual(vocab_utils.split(s), [s])  # Should not perform any vocab.split

    def testStrip(self):
        # Test case for ascii whitespace characters
        s = '\na*($*#(*$& b c    \r\n'
        self.assertAllEqual('a*($*#(*$& b c', vocab_utils.strip(s))

        # This string is not empty, it contains a non-ascii whitespace character. Run print(len(s)) to check
        s = u' '
        self.assertFalse(len(s) == 0)
        self.assertAllEqual(vocab_utils.strip(s), s)  # Should not remove any characters

    def testConvertToStr(self):
        # str input
        s = 'akf vna'
        result = vocab_utils.convert_to_str(s)
        self.assertTrue(isinstance(result, str))
        self.assertAllEqual(result, 'akf vna')

        # unicode input
        s = u'akf vna'
        result = vocab_utils.convert_to_str(s)
        self.assertTrue(isinstance(result, str))
        self.assertAllEqual(result, 'akf vna')

        # bytes input
        s = b'akf vna'
        result = vocab_utils.convert_to_str(s)
        self.assertTrue(isinstance(result, str))
        self.assertAllEqual(result, 'akf vna')

    def testConvertToBytes(self):
        # str input
        s = 'akf vna'
        result = vocab_utils.convert_to_bytes(s)
        self.assertTrue(isinstance(result, bytes))
        self.assertAllEqual(result, b'akf vna')

        # unicode input
        s = u'akf vna'
        result = vocab_utils.convert_to_bytes(s)
        self.assertTrue(isinstance(result, bytes))
        self.assertAllEqual(result, b'akf vna')

        # bytes input
        s = b'akf vna'
        result = vocab_utils.convert_to_bytes(s)
        self.assertTrue(isinstance(result, bytes))
        self.assertAllEqual(result, b'akf vna')

    def testConvertToUnicode(self):
        # str input
        s = 'akf vna'
        result = vocab_utils.convert_to_unicode(s)
        self.assertTrue(isinstance(result, six.text_type))
        self.assertAllEqual(result, u'akf vna')

        # unicode input
        s = u'akf vna'
        result = vocab_utils.convert_to_unicode(s)
        self.assertTrue(isinstance(result, six.text_type))
        self.assertAllEqual(result, u'akf vna')

        # bytes input
        s = b'akf vna'
        result = vocab_utils.convert_to_unicode(s)
        self.assertTrue(isinstance(result, six.text_type))
        self.assertAllEqual(result, u'akf vna')

    def testVocabLookUp(self):
        """Tests whether vocab lookup return the same result for str and unicode in python 2."""
        if six.PY2:
            # Switch to eager execution
            switch_to(EAGER_MODE)

            cur_dir = os.path.dirname(__file__)
            vocab_file = os.path.join(cur_dir, '..', 'resources', 'multilingual_vocab.txt.gz')
            vocab_table = vocab_utils.read_tf_vocab(vocab_file)

            def get_index(some_string):
                return vocab_table.lookup(tf.constant([some_string])).numpy()[0]

            unk = get_index(vocab_utils.UNK)

            # Non exist word
            self.assertEqual(get_index('bj82149aksreuo'), unk)

            # ascii characters
            s = 'a'
            s_u = u'a'

            self.assertAllEqual(get_index(s), get_index(s_u))
            self.assertTrue(get_index(s) != unk)

            # Special non-ascii characters
            s = '##' + '￥'
            s_u = '##' + u'￥'

            self.assertAllEqual(get_index(s), get_index(s_u))
            self.assertTrue(get_index(s) != unk)

            # Special non-ascii characters
            s = ' '
            s_u = u' '

            self.assertAllEqual(get_index(s), get_index(s_u))
            self.assertTrue(get_index(s) != unk)

            # Large-scale test
            # Disable this if you want fast build
            with gzip.GzipFile(fileobj=tf.gfile.Open(vocab_file, 'r')) as fin:
                for line in fin:
                    line = line.strip('\n')
                    if line != vocab_utils.UNK:
                        self.assertAllEqual(get_index(line), get_index(vocab_utils.convert_to_unicode(line)))
                        self.assertTrue(get_index(line) != unk)

            # Switch to graph execution
            switch_to(GRAPH_MODE)


if __name__ == "__main__":
    tf.test.main()
