"""
Utility function for vocabulary
"""
import gzip
import os
import re
import string

import six
import tensorflow as tf
from tensorflow.python.ops import lookup_ops


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        # In py2, six.text_type = unicode. We use six.text_type to pass the pyflake checking
        elif isinstance(text, six.text_type):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def strip(s):
    """Strips ascii whitespace characters off string s."""
    return s.strip(string.whitespace)


def split(s):
    """Split string s by whitespace characters."""
    whitespace_lst = [re.escape(ws) for ws in string.whitespace]
    pattern = re.compile('|'.join(whitespace_lst))
    return pattern.split(s)


def read_vocab(input_file):
    """Read vocabulary file and return a dict"""
    if input_file is None:
        return None

    vocab = {}
    if input_file.endswith('.gz'):
        f = tf.io.gfile.GFile(input_file, 'r')
        fin = gzip.GzipFile(fileobj=f)
    else:
        fin = tf.io.gfile.GFile(input_file, 'r')
    for line in fin:
        word = split(strip(line))[0]
        vocab[word] = len(vocab)
    fin.close()
    return vocab


def read_tf_vocab(input_file, UNK):
    """Read vocabulary and return a tf hashtable"""
    if input_file is None:
        return None

    keys, values = [], []
    if input_file.endswith('.gz'):
        f = tf.io.gfile.GFile(input_file, 'r')
        fin = gzip.GzipFile(fileobj=f)
    else:
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


def extract_text_data(input_dir, output_file, text_fields):
    """
    Extract text data from tfrecords.  The data will be used for word embedding pretraining.
    """
    with tf.io.gfile.GFile(output_file, 'w') as fout:
        for file in tf.io.gfile.listdir(input_dir):
            input_file = os.path.join(input_dir, file)
            print(input_file)
            for example in tf.compat.v1.python_io.tf_record_iterator(input_file):
                result = tf.train.Example.FromString(example)
                for field in text_fields:
                    text_values = result.features.feature[field].bytes_list.value
                    for text in text_values:
                        text = convert_to_unicode(text)
                        text = text.strip()
                        if ' ' not in text:
                            continue
                        fout.write(text + '\n')
