"""
Utility function for vocabulary
"""
import gzip
import os
import re
import six
import string
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


def convert_to_str(text):
    """
    Converts `text` to str (if it's not already).
    Used when writing text to file with `open(filename, 'w')`.
    """
    if six.PY2:  # In python 2, str is byte
        if isinstance(text, str):
            return text
        else:
            return convert_to_unicode(text).encode('utf-8')
    elif six.PY3:  # In python3, str is unicode
        if isinstance(text, str):
            return text
        else:
            return convert_to_unicode(text)
    raise ValueError("Not running on Python2 or Python 3?")


def convert_to_bytes(text):
    """
    Converts `text` to bytes (if it's not already).
    Used when generating tfrecords. More specifically, in function call `tf.train.BytesList(value=[<bytes1>, <bytes2>, ...])`
    """
    if six.PY2:
        return convert_to_str(text)  # In python2, str is byte
    elif six.PY3:
        if isinstance(text, bytes):
            return text
        else:
            return convert_to_unicode(text).encode('utf-8')
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
        f = tf.gfile.Open(input_file, 'r')
        fin = gzip.GzipFile(fileobj=f)
    else:
        fin = tf.gfile.Open(input_file, 'r')
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
        f = tf.gfile.Open(input_file, 'r')
        fin = gzip.GzipFile(fileobj=f)
    else:
        fin = tf.gfile.Open(input_file, 'r')
    for line in fin:
        word = split(strip(line))[0]
        keys.append(word)
        values.append(len(values))
    fin.close()
    UNK_ID = keys.index(UNK)
    vocab_table = lookup_ops.HashTable(lookup_ops.KeyValueTensorInitializer(
        tf.constant(keys), tf.constant(values)), UNK_ID)
    return vocab_table


def extract_text_data(input_dir, output_file, text_fields):
    """
    Extract text data from tfrecords.  The data will be used for word embedding pretraining.
    """
    with tf.gfile.Open(output_file, 'w') as fout:
        for file in tf.gfile.ListDirectory(input_dir):
            input_file = os.path.join(input_dir, file)
            print(input_file)
            for example in tf.python_io.tf_record_iterator(input_file):
                result = tf.train.Example.FromString(example)
                for field in text_fields:
                    text_values = result.features.feature[field].bytes_list.value
                    for text in text_values:
                        text = convert_to_unicode(text)
                        text = text.strip()
                        if ' ' not in text:
                            continue
                        fout.write(text + '\n')


if __name__ == '__main__':
    input_dir = '/home/wguo/data/people_v6/NAV/train/'
    output_file = '/home/wguo/data/people_v6/NAV/glove/text'
    text_fields = ['query', 'doc_headlines', 'doc_currCompanies', 'doc_pastCompanies', 'doc_currTitles',
                   'doc_pastTitles', 'doc_currSchools', 'doc_pastSchools']
    extract_text_data(input_dir, output_file, text_fields)
    # max_vocab_size = 100000
    # generate_vocab_file(input_dir, output_file, max_vocab_size)
