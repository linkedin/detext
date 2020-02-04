"""Utility functions for building models."""

import tensorflow as tf

from detext.train import data_fn
from detext.utils import vocab_utils


def create_placeholder_for_ftrs(ph_name, shape, dtype, ftr_name, ftr_names):
    """Creates placeholder and specifies batch_size=1"""
    ph = tf.placeholder(shape=shape, dtype=dtype, name=ph_name)
    ph_one_batch = tf.expand_dims(ph, axis=0) if ftr_name in ftr_names else None
    return ph, ph_one_batch


def get_query(hparams, regex_replace_pattern, add_dimension=False):
    """
    Helper function to get query and query_placeholder
    :param hparams: hparams
    :param regex_replace_pattern: The regex pattern to add a white space before and after
    :param add_dimension: whether to add a dimension then remove to query (this is to support online model for QAP as
    quasar model serving requires at least one dimension)
    :return: query and query_placeholder
    """
    # query text feature
    if add_dimension:
        query_placeholder, query = create_placeholder_for_ftrs("query_placeholder", [None], tf.string, 'query',
                                                               hparams.feature_names)
    else:
        query_placeholder, query = create_placeholder_for_ftrs("query_placeholder", [], tf.string, 'query',
                                                               hparams.feature_names)
    if query is not None:
        if add_dimension:
            # remove added dimension
            query = tf.squeeze(query, [0])

        # tokenize query
        if regex_replace_pattern is not None:
            query = tf.regex_replace(input=query, pattern=regex_replace_pattern, rewrite=" \\1 ")

        query = data_fn.process_text(
            query,
            vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
            hparams.CLS, hparams.SEP, hparams.PAD,
            hparams.max_len,
            hparams.min_len,
            cnn_filter_window_size=max(hparams.filter_window_sizes)
        )
    return query, query_placeholder


def get_doc_fields(hparams, regex_replace_pattern):
    """
    Each document field has a placeholder.
    The regex is to add whitespace on both sides of punctuations.
    :param hparams: hparams
    :param regex_replace_pattern: The regex pattern to add a white space before and after
    :return:
    """
    doc_text_placeholders = []
    doc_fields = []
    tf_vocab_table = vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK)
    for ftr_name in hparams.feature_names:
        if ftr_name.startswith('doc_'):
            # each document field is a placeholder (a string vector)
            placeholder = tf.placeholder(shape=[None], dtype=tf.string, name=ftr_name + "_placeholder")
            doc_text_placeholders.append(placeholder)

            one_doc_field = placeholder
            # add whitespace on both sides of punctuations if regex pattern is not None
            if regex_replace_pattern is not None:
                one_doc_field = tf.regex_replace(input=one_doc_field, pattern=regex_replace_pattern, rewrite=" \\1 ")
            one_doc_field = data_fn.process_text(
                one_doc_field,
                tf_vocab_table,
                hparams.CLS, hparams.SEP, hparams.PAD,
                hparams.max_len,
                hparams.min_len,
                cnn_filter_window_size=max(hparams.filter_window_sizes)
            )
            one_doc_field = tf.expand_dims(one_doc_field, axis=0)
            doc_fields.append(one_doc_field)
    return doc_fields, doc_text_placeholders


def get_usr_fields(hparams, regex_replace_pattern):
    """
    Each user field has a placeholder.
    The regex is to add whitespace on both sides of punctuations.
    :param hparams: hparams
    :param regex_replace_pattern: The regex pattern to add a white space before and after
    :return:
    """
    usr_text_placeholders = []
    usr_fields = []
    tf_vocab_table = vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK)
    for ftr_name in hparams.feature_names:
        if ftr_name.startswith('usr_'):
            # each user field is a placeholder (one string)
            placeholder = tf.placeholder(shape=[], dtype=tf.string, name=ftr_name + "_placeholder")
            usr_text_placeholders.append(placeholder)

            one_usr_field = placeholder
            # add whitespace on both sides of punctuations if regex pattern is not None
            if regex_replace_pattern is not None:
                one_usr_field = tf.regex_replace(input=one_usr_field, pattern=regex_replace_pattern, rewrite=" \\1 ")

            one_usr_field = tf.expand_dims(one_usr_field, axis=0)
            one_usr_field = data_fn.process_text(
                one_usr_field,
                tf_vocab_table,
                hparams.CLS, hparams.SEP, hparams.PAD,
                hparams.max_len,
                hparams.min_len,
                cnn_filter_window_size=max(hparams.filter_window_sizes)
            )
            usr_fields.append(one_usr_field)
    return usr_fields, usr_text_placeholders
