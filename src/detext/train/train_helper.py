"""Utility functions for building models."""

import tensorflow as tf

from detext.train import data_fn
from detext.utils import vocab_utils


def create_placeholder_for_ftrs(ph_name, shape, dtype, ftr_name, ftr_names, default=None):
    """Creates placeholder and specifies batch_size=1"""
    ph = tf.placeholder(shape=shape, dtype=dtype, name=ph_name)
    ph_one_batch = tf.expand_dims(ph, axis=0) if ftr_name in ftr_names else default
    return ph, ph_one_batch


def get_query(hparams):
    """
    Helper function to get query and query_placeholder
    :param hparams: hparams
    :return: query and query_placeholder
    """
    # query text feature
    # If hparams.add_first_dim_for_query_placeholder is True, the query placeholder has dimension [None]
    # This is to use the query feature as a document feature in model serving
    if hparams.add_first_dim_for_query_placeholder:
        query_placeholder, query = create_placeholder_for_ftrs("query_placeholder", [None], tf.string, 'query',
                                                               hparams.feature_names)
    else:
        query_placeholder, query = create_placeholder_for_ftrs("query_placeholder", [], tf.string, 'query',
                                                               hparams.feature_names)
    if query is not None:
        if hparams.add_first_dim_for_query_placeholder:
            # remove added dimension
            query = tf.squeeze(query, [0])

        # tokenize query
        if hparams.regex_replace_pattern is not None:
            query = tf.regex_replace(input=query, pattern=hparams.regex_replace_pattern, rewrite=" \\1 ")

        query = data_fn.process_text(
            query,
            vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK),
            hparams.CLS, hparams.SEP, hparams.PAD,
            hparams.max_len,
            hparams.min_len,
            cnn_filter_window_size=max(hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
        )
    return query, query_placeholder


def get_doc_fields(hparams):
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
            if hparams.regex_replace_pattern is not None:
                one_doc_field = tf.regex_replace(input=one_doc_field, pattern=hparams.regex_replace_pattern, rewrite=" \\1 ")
            one_doc_field = data_fn.process_text(
                one_doc_field,
                tf_vocab_table,
                hparams.CLS, hparams.SEP, hparams.PAD,
                hparams.max_len,
                hparams.min_len,
                cnn_filter_window_size=max(hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
            )
            one_doc_field = tf.expand_dims(one_doc_field, axis=0)
            doc_fields.append(one_doc_field)
    return doc_fields, doc_text_placeholders


def get_usr_fields(hparams):
    """
    Each user field has a placeholder.
    The regex is to add whitespace on both sides of punctuations.
    :param hparams: hparams
    :return:
    """
    usr_text_placeholders = []
    usr_fields = []
    tf_vocab_table = vocab_utils.read_tf_vocab(hparams.vocab_file, hparams.UNK)
    for ftr_name in hparams.feature_names:
        if ftr_name.startswith('usr_'):
            # If hparams.add_first_dim_for_usr_placeholder is True, the usr placeholders have dimension [None]
            # This is to use the usr field features as document features in model serving
            if hparams.add_first_dim_for_usr_placeholder:
                # each user field is a placeholder (one string)
                placeholder = tf.placeholder(shape=[None], dtype=tf.string, name=ftr_name + "_placeholder")
            else:
                placeholder = tf.placeholder(shape=[], dtype=tf.string, name=ftr_name + "_placeholder")
            usr_text_placeholders.append(placeholder)

            one_usr_field = placeholder
            # add whitespace on both sides of punctuations if regex pattern is not None
            if hparams.regex_replace_pattern is not None:
                one_usr_field = tf.regex_replace(input=one_usr_field, pattern=hparams.regex_replace_pattern, rewrite=" \\1 ")

            # remove added dimension
            if hparams.add_first_dim_for_usr_placeholder:
                one_usr_field = tf.squeeze(one_usr_field, [0])
            one_usr_field = tf.expand_dims(one_usr_field, axis=0)
            one_usr_field = data_fn.process_text(
                one_usr_field,
                tf_vocab_table,
                hparams.CLS, hparams.SEP, hparams.PAD,
                hparams.max_len,
                hparams.min_len,
                cnn_filter_window_size=max(hparams.filter_window_sizes) if hparams.ftr_ext == 'cnn' else 0
            )
            usr_fields.append(one_usr_field)
    return usr_fields, usr_text_placeholders


def get_doc_id_fields(hparams):
    """ Returns a list of processed doc id fields and the corresponding list of raw doc id placeholders
    Each document id field has a placeholder
    """
    doc_id_placeholders = []
    doc_fields = []
    tf_vocab_table = vocab_utils.read_tf_vocab(hparams.vocab_file_for_id_ftr, hparams.UNK_FOR_ID_FTR)
    for ftr_name in hparams.feature_names:
        if ftr_name.startswith('docId_'):
            # each document id field is a placeholder (a string vector)
            placeholder = tf.placeholder(shape=[None], dtype=tf.string, name=ftr_name + "_placeholder")
            doc_id_placeholders.append(placeholder)

            one_doc_field = placeholder
            one_doc_field = data_fn.process_id(
                one_doc_field,
                tf_vocab_table,
                hparams.PAD_FOR_ID_FTR
            )
            one_doc_field = tf.expand_dims(one_doc_field, axis=0)
            doc_fields.append(one_doc_field)
    return doc_fields, doc_id_placeholders


def get_usr_id_fields(hparams):
    """ Returns a list of processed usr id fields and the corresponding list of raw usr id placeholders
    Each user field has a placeholder
    """
    usr_id_placeholders = []
    usr_fields = []
    tf_vocab_table = vocab_utils.read_tf_vocab(hparams.vocab_file_for_id_ftr, hparams.UNK_FOR_ID_FTR)
    for ftr_name in hparams.feature_names:
        if ftr_name.startswith('usrId_'):
            # each user id field is a placeholder (one string)
            placeholder = tf.placeholder(shape=[], dtype=tf.string, name=ftr_name + "_placeholder")
            usr_id_placeholders.append(placeholder)

            one_usr_field = placeholder

            one_usr_field = tf.expand_dims(one_usr_field, axis=0)
            one_usr_field = data_fn.process_id(
                one_usr_field,
                tf_vocab_table,
                hparams.PAD_FOR_ID_FTR
            )

            usr_fields.append(one_usr_field)
    return usr_fields, usr_id_placeholders
