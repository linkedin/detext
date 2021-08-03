import pickle

import tensorflow as tf
from absl import logging


def init_word_embedding(vocab_size, num_units, we_trainable, we_file=None, name_prefix="w"):
    """Initialize word embeddings from random initialization or pretrained word embedding.

    This function is only used by encoding models other than BERT
    """

    if not we_file:
        embedding_name = "{}_pretrained_embedding".format(name_prefix)
        # Random initialization
        embedding = tf.compat.v1.get_variable(
            embedding_name, [vocab_size, num_units], dtype=tf.float32, trainable=we_trainable)
        logging.info(f'Initializing embedding {embedding_name}')
    else:
        # Initialize by pretrained word embedding
        embedding_name = "{}_embedding".format(name_prefix)
        we = pickle.load(tf.io.gfile.GFile(we_file, 'rb'))
        assert vocab_size == we.shape[0] and num_units == we.shape[1]
        embedding = tf.compat.v1.get_variable(name=embedding_name,
                                              shape=[vocab_size, num_units],
                                              dtype=tf.float32,
                                              initializer=tf.compat.v1.constant_initializer(we),
                                              trainable=we_trainable)
        logging.info(f'Loading pretrained embedding {embedding_name} from {we_file}')
    return embedding


def get_sorted_dict(dct: dict):
    """Returns dictionary in sorted order"""
    return dict(sorted(dct.items()))
