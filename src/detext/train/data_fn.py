"""
Data processing.  Two important functions:
1. training_input_fn(), used during training and evaluation.
2. serving_input_receiver_fn(), used by model serving.
"""

import tensorflow as tf
import tensorflow_ranking as tfr

from detext.utils.misc_utils import get_input_files


def process_text(text,
                 vocab_table,
                 CLS, SEP, PAD,
                 max_len=None,
                 min_len=None,
                 cnn_filter_window_size=0):
    """
    Text processing: add cls/sep, handle max_len/min_len, etc.
    text must be a string vector.
    """
    # add cls and sep
    num_docs = tf.shape(text)[0]
    cls_vec = tf.tile([CLS], [num_docs])
    sep_vec = tf.tile([SEP], [num_docs])
    if cnn_filter_window_size <= 2:
        text = tf.reduce_join([cls_vec, text, sep_vec], 0, separator=' ')
    else:
        # make sure each word appears the same times in CNN filters
        to_join = []
        for _ in range(cnn_filter_window_size - 1):
            to_join.append(cls_vec)
        to_join.append(text)
        for _ in range(cnn_filter_window_size - 1):
            to_join.append(sep_vec)
        text = tf.strings.reduce_join(to_join, 0, separator=' ')

    # Split by spaces, and generate dense matrix
    text_sp = tf.string_split(text)
    text = tf.sparse.to_dense(text_sp, default_value=PAD)

    # handle max_len
    if max_len:
        text = text[:, :max_len]
    # handle min_len
    if min_len:
        text = tf.cond(
            tf.shape(text)[1] >= min_len,
            lambda: text,
            lambda: tf.pad(text, [[0, 0], [0, min_len - tf.shape(text)[1]]], constant_values=PAD))

    # Convert the word strings to ids.
    text = tf.cast(vocab_table.lookup(text), tf.int32)

    return text


def input_fn(input_pattern,
             batch_size,
             mode,
             vocab_table,
             feature_names,
             CLS, SEP, PAD,
             max_len=None,
             min_len=None,
             cnn_filter_window_size=0,
             block_length=100,
             prefetch_size=10):
    """
    Data input function for training given TFRecord
    """
    output_buffer_size = batch_size * 1000

    input_files = get_input_files(input_pattern)
    feature_names = list(feature_names)
    if len(input_files) > 1:  # Multiple input files
        # Preprocess files concurrently, and interleave blocks of block_length records from each file.
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=len(input_files),
                                     block_length=block_length)
    else:
        dataset = tf.data.TFRecordDataset(input_files[0])

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(output_buffer_size)
        dataset = dataset.repeat()

    def process_data(record):
        # parse tfrecord
        example = tf.io.parse_single_example(record, features_tfr)

        # process each field
        example2 = dict()
        for name in feature_names:
            t = example[name]
            # change sparse data to dense matrix
            if name == 'wide_ftrs_sp_val':
                t = tf.sparse.to_dense(t, default_value=1.)  # For sparse features values, padding should be 1
            elif name != 'query' and not name.startswith('usr_'):
                if t.dtype == tf.string:
                    t = tf.sparse.to_dense(t, default_value='')
                else:
                    # If there's sparse wide features, it's required that 0 is only used for
                    #   padding (not for feature representation)
                    t = tf.sparse.to_dense(t, default_value=0)

            # use tf.int32 instead of tf.int64
            if t.dtype == tf.int64:
                t = tf.to_int32(t)

            # for text data, call process_text()
            if name == 'query' or name.startswith('doc_') or name.startswith('usr_'):
                t = process_text(t, vocab_table, CLS, SEP, PAD, max_len, min_len, cnn_filter_window_size)
            if name == 'query' or name.startswith('usr_'):
                t = tf.squeeze(t, axis=0)
            example2[name] = t

        example = example2

        num_docs = tf.size(example['label'])
        # get group size from label
        example2['group_size'] = num_docs
        if 'wide_ftrs' in feature_names:
            example['wide_ftrs'] = tf.reshape(example['wide_ftrs'], shape=[num_docs, -1])
        if 'wide_ftrs_sp_idx' in feature_names:
            example['wide_ftrs_sp_idx'] = tf.reshape(example['wide_ftrs_sp_idx'], shape=[num_docs, -1])
        if 'wide_ftrs_sp_val' in feature_names:
            example['wide_ftrs_sp_val'] = tf.reshape(example['wide_ftrs_sp_val'], shape=[num_docs, -1])

        label = {'label': example.pop('label')}
        return example, label

    # Generate features for parsing
    features_tfr = dict()
    for name in feature_names:
        if name == 'query' or name.startswith('usr_'):
            features_tfr[name] = tf.io.FixedLenFeature(shape=[1], dtype=tf.string)
        elif name in ('wide_ftrs', 'label', 'wide_ftrs_sp_val'):
            features_tfr[name] = tf.io.VarLenFeature(dtype=tf.float32)
        elif name in ('wide_ftrs_sp_idx',):
            features_tfr[name] = tf.io.VarLenFeature(dtype=tf.int64)
        else:
            assert name.startswith('doc_'), \
                "Except query/wide_ftrs/label/usr_xxx, feature names must start with doc_: %s" % name
            features_tfr[name] = tf.io.VarLenFeature(dtype=tf.string)

    # process data
    dataset = dataset.map(lambda record: process_data(record))

    # Pad
    padded_shapes = {'group_size': tf.TensorShape([])}
    padded_values = {'group_size': 0}

    feature_names.remove('label')

    pad_id = tf.cast(vocab_table.lookup(tf.constant(PAD)), tf.int32)
    for name in feature_names:
        if name == 'query' or name.startswith('usr_'):
            padded_shapes[name] = tf.TensorShape([None])
            padded_values[name] = pad_id
        elif name == 'wide_ftrs':
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = 0.0
        elif name == 'wide_ftrs_sp_val':
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = 1.0
        elif name == 'wide_ftrs_sp_idx':
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = 0
        elif name.startswith('doc_'):
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = pad_id
        else:
            raise KeyError('Feature name ({}) not supported'.format(name))

    padded_shapes = (padded_shapes, {'label': tf.TensorShape([None])})
    padded_values = (padded_values, {'label': tfr.data._PADDING_LABEL})  # use -1 as padding for labels
    dataset = (dataset
               .padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padded_values)
               .prefetch(prefetch_size))
    return dataset
