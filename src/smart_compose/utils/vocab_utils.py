""" Vocabulary construction utilities """
import gzip
import re
import string

import tensorflow as tf
from tensorflow.python.ops import lookup_ops


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


def read_tf_vocab_inverse(input_file, UNK):
    """Read vocabulary (token->id) and return a tf hashtable (id->token)"""
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

        keys.append(len(keys))
        values.append(word)
    fin.close()

    initializer = lookup_ops.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    vocab_table = lookup_ops.HashTable(initializer, UNK)
    return initializer, vocab_table


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
