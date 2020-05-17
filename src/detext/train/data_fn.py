"""
Data processing.  Two important functions:
1. training_input_fn(), used during training and evaluation.
2. serving_input_receiver_fn(), used by model serving.
"""

import tensorflow as tf
import tensorflow_ranking as tfr
from functools import partial

from detext.utils.misc_utils import get_input_files

_LABEL_PADDING = tfr.data._PADDING_LABEL


def _get_padded_shapes_and_values(feature_names, PAD, vocab_table, PAD_FOR_ID_FTR, vocab_table_for_id_ftr):
    """Returns padded_shape and padd_values for each feature in feature_names

    :param feature_names List of feature names EXCLUDING 'label'
    :param pad_id Id of padding token in the vocab table. Used as the padding value for text fields.
    """
    padded_shapes = {'group_size': tf.TensorShape([])}
    padded_values = {'group_size': 0}

    pad_id = tf.cast(vocab_table.lookup(tf.constant(PAD)), tf.int32)

    if vocab_table_for_id_ftr is not None:
        pad_id_for_id_ftr = tf.cast(vocab_table_for_id_ftr.lookup(tf.constant(PAD_FOR_ID_FTR)), tf.int32)

    for name in feature_names:
        if name == 'query' or name.startswith('usr_'):
            padded_shapes[name] = tf.TensorShape([None])
            padded_values[name] = pad_id
        elif name.startswith('doc_'):
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = pad_id
        elif name.startswith('usrId_'):
            assert vocab_table_for_id_ftr is not None, "Need to provide vocab_table_for_id_ftr"
            padded_shapes[name] = tf.TensorShape([None])
            padded_values[name] = pad_id_for_id_ftr
        elif name.startswith('docId_'):
            assert vocab_table_for_id_ftr is not None, "Need to provide vocab_table_for_id_ftr"
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = pad_id_for_id_ftr
        elif name == 'weight':
            padded_shapes[name] = tf.TensorShape([])
            padded_values[name] = 0.0
        elif name in ('uid', 'task_id'):
            padded_shapes[name] = tf.TensorShape([])
            padded_values[name] = tf.cast(0, tf.int64)
        elif name == 'wide_ftrs':
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = 0.0
        elif name == 'wide_ftrs_sp_val':
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = 1.0
        elif name == 'wide_ftrs_sp_idx':
            padded_shapes[name] = tf.TensorShape([None, None])
            padded_values[name] = 0
        elif name == 'label':
            padded_shapes[name] = tf.TensorShape([None])
            padded_values[name] = _LABEL_PADDING
        else:
            raise KeyError('Feature name ({}) not supported'.format(name))

    padded_shapes = (padded_shapes, {'label': tf.TensorShape([None])})
    padded_values = (padded_values, {'label': _LABEL_PADDING})  # use -1 as padding for labels
    return padded_shapes, padded_values


def _get_tfrecord_feature_parsing_schema(feature_names):
    """Returns parsing schema for input TFRecord

    :param feature_names: List of feature names INCLUDING 'label'
    """
    features_tfr = dict()
    for name in feature_names:
        if name == 'query' or name.startswith('usr_') or name.startswith('usrId_'):
            features_tfr[name] = tf.FixedLenFeature(shape=[1], dtype=tf.string)
        elif name == 'weight':
            features_tfr[name] = tf.FixedLenFeature(shape=[1], dtype=tf.float32)
            # Default uid as feature for detext integration, will be -1 by default if not present in data
        elif name in ('uid', 'task_id'):
            features_tfr[name] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
        elif name.startswith(('doc_', 'docId_')):
            features_tfr[name] = tf.VarLenFeature(dtype=tf.string)
        elif name in ('wide_ftrs', 'label', 'wide_ftrs_sp_val'):
            features_tfr[name] = tf.VarLenFeature(dtype=tf.float32)
        elif name in ('wide_ftrs_sp_idx',):
            features_tfr[name] = tf.VarLenFeature(dtype=tf.int64)
        else:
            raise KeyError("Unknown feature in tfrecord feature parsing: %s" % name)
    return features_tfr


def _cast_to_dtype_of_smaller_size(t):
    """Casts tensor to smaller storage dtype. int64 -> int32, float64 -> float32"""
    if t.dtype == tf.int64:
        return tf.to_int32(t)
    elif t.dtype == tf.float64:
        return tf.to_float(t)
    else:
        return t


def _convert_ftrs_to_dense_tensor(t, name):
    """Converts the VarLenFeature to dense format"""
    if name == 'wide_ftrs_sp_val':
        t = tf.sparse_tensor_to_dense(t, default_value=1.)  # For sparse features values, padding should be 1
    elif name.startswith('doc_') or name.startswith('docId_'):
        t = tf.sparse_tensor_to_dense(t, default_value='')
    elif name == 'wide_ftrs':
        t = tf.sparse_tensor_to_dense(t, default_value=0.)
    elif name == 'label':
        t = tf.sparse_tensor_to_dense(t, default_value=_LABEL_PADDING)
    elif name == 'wide_ftrs_sp_idx':  # If there's sparse wide features, it's required that 0 is only used for padding
        t = tf.sparse_tensor_to_dense(t, default_value=0)
    else:
        pass
    return t


def _reshape_ftrs_to_group_wise(features, feature_names, num_docs):
    """Reshapes features from [num_fts] to [num_ftrs_each_doc, num_docs]

    :param features Mapping from feature name to tensor
    """
    if 'wide_ftrs' in feature_names:
        features['wide_ftrs'] = tf.reshape(features['wide_ftrs'], shape=[num_docs, -1])
    if 'wide_ftrs_sp_idx' in feature_names:
        features['wide_ftrs_sp_idx'] = tf.reshape(features['wide_ftrs_sp_idx'], shape=[num_docs, -1])
    if 'wide_ftrs_sp_val' in feature_names:
        features['wide_ftrs_sp_val'] = tf.reshape(features['wide_ftrs_sp_val'], shape=[num_docs, -1])
    return features


def process_text(text,
                 vocab_table,
                 CLS, SEP, PAD,
                 max_len=None,
                 min_len=None,
                 cnn_filter_window_size=0):
    """
    Text processing: add cls/sep, handle max_len/min_len, etc.
    Text must be a string vector.
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
        text = tf.reduce_join(to_join, 0, separator=' ')

    # Split by spaces, and generate dense matrix
    text_sp = tf.string_split(text)
    text = tf.sparse_tensor_to_dense(text_sp, default_value=PAD)

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


def process_id(input_ids,
               vocab_table,
               PAD,
               max_len=None,
               min_len=None):
    """Converts input ids (in text string) into their indices in vocab

    :param input_ids Tensor Id features. Shape=[group_size]
    :param vocab_table TFLookupTable Vocab table for id features
    :param PAD Padding token for id features
    :param max_len Maximum number of id features for one doc
    :param min_len Minimum number of id features for one doc
    """
    # Split by spaces, and generate dense matrix
    input_ids_sp = tf.string_split(input_ids)
    input_ids = tf.sparse_tensor_to_dense(input_ids_sp, default_value=PAD)

    # handle max_len
    if max_len:
        input_ids = input_ids[:, :max_len]

    # handle min_len
    if min_len:
        input_ids = tf.cond(
            tf.shape(input_ids)[1] >= min_len,
            lambda: input_ids,
            lambda: tf.pad(input_ids, [[0, 0], [0, min_len - tf.shape(input_ids)[1]]], constant_values=PAD))

    # Convert input id strings to indices
    input_ids = tf.cast(vocab_table.lookup(input_ids), tf.int32)

    return input_ids


def input_fn(input_pattern, metadata_path, batch_size, mode,
             vocab_table, vocab_table_for_id_ftr,
             feature_names,
             CLS, SEP, PAD, PAD_FOR_ID_FTR,
             max_len=None,
             min_len=None,
             cnn_filter_window_size=0,
             prefetch_size=100,
             block_length=100,
             hvd_info=None):
    """ Returns a dataset from avro/tfrecord file """
    return input_fn_tfrecord(input_pattern, batch_size, mode,
                             vocab_table, vocab_table_for_id_ftr,
                             feature_names,
                             CLS, SEP, PAD, PAD_FOR_ID_FTR,
                             max_len, min_len,
                             cnn_filter_window_size,
                             block_length,
                             prefetch_size,
                             hvd_info=hvd_info)


def input_fn_tfrecord(input_pattern,
                      batch_size,
                      mode,
                      vocab_table, vocab_table_for_id_ftr,
                      feature_names,
                      CLS, SEP, PAD, PAD_FOR_ID_FTR,
                      max_len=None,
                      min_len=None,
                      cnn_filter_window_size=0,
                      block_length=100,
                      prefetch_size=100,
                      num_data_process_threads=32,
                      hvd_info=None):
    """
    Data input function for training given TFRecord
    """
    output_buffer_size = 1000

    input_files = get_input_files(input_pattern)
    feature_names = list(feature_names)
    if len(input_files) > 1:  # Multiple input files
        # Preprocess files concurrently, and interleave blocks of block_length records from each file
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        # shard input when using horovod
        if hvd_info is not None:
            dataset = dataset.shard(hvd_info['size'], hvd_info['rank'])
        dataset = dataset.shuffle(buffer_size=len(input_files))
        hvd_size = hvd_info['size'] if hvd_info is not None else 1
        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_data_process_threads, int(len(input_files) / hvd_size))
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_length,
                                     block_length=block_length)
    else:
        dataset = tf.data.TFRecordDataset(input_files[0])

    dataset = tfrecord_transform_fn(dataset,
                                    batch_size,
                                    mode,
                                    vocab_table, vocab_table_for_id_ftr,
                                    feature_names,
                                    CLS, SEP, PAD, PAD_FOR_ID_FTR,
                                    output_buffer_size,
                                    max_len,
                                    min_len,
                                    cnn_filter_window_size,
                                    prefetch_size,
                                    num_data_process_threads)
    return dataset


def tfrecord_transform_fn(dataset,
                          batch_size,
                          mode,
                          vocab_table, vocab_table_for_id_ftr,
                          feature_names,
                          CLS, SEP, PAD, PAD_FOR_ID_FTR,
                          output_buffer_size,
                          max_len=None,
                          min_len=None,
                          cnn_filter_window_size=0,
                          prefetch_size=100,
                          num_data_process_threads=32):
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(output_buffer_size)
        dataset = dataset.repeat()

    def process_data(record, features_schema):
        # parse tfrecord
        example = tf.parse_single_example(record, features_schema)

        # process each field
        features = dict()
        for name in feature_names:
            t = example[name]
            if name not in ('uid', 'task_id'):
                t = _cast_to_dtype_of_smaller_size(t)
            t = _convert_ftrs_to_dense_tensor(t, name)

            # for text data, call process_text()
            if name == 'query' or name.startswith('doc_') or name.startswith('usr_'):
                t = process_text(t, vocab_table, CLS, SEP, PAD, max_len, min_len, cnn_filter_window_size)

            if name.startswith('usrId_') or name.startswith('docId_'):
                t = process_id(t, vocab_table_for_id_ftr, PAD_FOR_ID_FTR)

            if name in ('query', 'weight', 'uid', 'task_id') or name.startswith(('usr_', 'usrId_')):
                t = tf.squeeze(t, axis=0)

            features[name] = t

        num_docs = tf.size(features['label'])
        # get group size from label
        features['group_size'] = num_docs
        features = _reshape_ftrs_to_group_wise(features, feature_names, num_docs)
        features.setdefault('weight', tf.constant(1.0, dtype=tf.float32))
        # Default uid as feature for detext integration, will be -1 by default if not present in data
        features.setdefault('uid', tf.constant(-1, dtype=tf.int64))

        label = {'label': tf.to_float(features['label'])}
        return features, label

    features_schema = _get_tfrecord_feature_parsing_schema(feature_names)
    # process data
    dataset = dataset.map(partial(process_data, features_schema=features_schema),
                          num_parallel_calls=num_data_process_threads)

    if 'weight' not in feature_names:
        feature_names.append('weight')
    if 'uid' not in feature_names:
        feature_names.append('uid')
    padded_shapes, padded_values = _get_padded_shapes_and_values(feature_names, PAD, vocab_table,
                                                                 PAD_FOR_ID_FTR, vocab_table_for_id_ftr)

    dataset = (dataset
               .padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padded_values)
               .prefetch(prefetch_size))
    return dataset
