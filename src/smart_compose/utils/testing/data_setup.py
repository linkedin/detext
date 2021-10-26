import os

import tensorflow as tf

from smart_compose.utils.vocab_utils import read_vocab


class DataSetup:
    """Class containing common setup on file paths, layer params used in unit tests"""
    resource_dir = os.path.join(os.getcwd(), 'test', 'smart_compose', 'resources')
    out_dir = os.path.join(resource_dir, "output")
    data_dir = os.path.join(resource_dir, "train", "dataset", "tfrecord")

    # Vocab
    vocab_file = os.path.join(resource_dir, 'vocab.txt')
    large_vocab_file = os.path.join(resource_dir, 'vocab.30k.txt')
    vocab_layer_dir = os.path.join(resource_dir, 'vocab_layer')
    vocab_hub_url = os.path.join(resource_dir, 'vocab_layer_hub')
    vocab_table_py = read_vocab(vocab_file)
    vocab_size = len(vocab_table_py)

    # Embedding layer
    we_file = ''
    embedding_layer_dir = os.path.join(resource_dir, 'embedding_layer')
    embedding_hub_url = os.path.join(resource_dir, 'embedding_layer_hub')

    # Special tokens
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    UNK = '[UNK]'
    CLS_ID = vocab_table_py[CLS]
    PAD_ID = vocab_table_py[PAD]
    SEP_ID = vocab_table_py[SEP]
    UNK_ID = vocab_table_py[UNK]

    # Vocab layer
    vocab_layer_param = {'CLS': CLS,
                         'SEP': SEP,
                         'PAD': PAD,
                         'UNK': UNK,
                         'vocab_file': vocab_file}

    # Embedding layer
    num_units = 10
    embedding_layer_param = {'vocab_layer_param': vocab_layer_param,
                             'vocab_hub_url': '',
                             'we_file': '',
                             'we_trainable': True,
                             'num_units': num_units}

    # Vocab layer with larger vocabulary size
    large_vocab_layer_param = {'CLS': CLS,
                               'SEP': SEP,
                               'PAD': PAD,
                               'UNK': UNK,
                               'vocab_file': large_vocab_file}

    # Embedding layer with larger embedding size
    num_units_large = 200
    embedding_layer_with_large_vocab_layer_param = {'vocab_layer_param': large_vocab_layer_param,
                                                    'vocab_hub_url': '',
                                                    'we_file': '',
                                                    'we_trainable': True,
                                                    'num_units': num_units_large}

    empty_url = ''

    # Beam search params
    min_len = 3
    max_len = 7
    beam_width = 10
    max_iter = 3
    max_decode_length = 3
    min_seq_prob = 0.01
    length_norm_power = 0

    # Target testing
    target_text = tf.constant(['test', 'function', 'hello'], dtype=tf.dtypes.string)
