import tensorflow as tf
from functools import partial

from smart_compose.utils.parsing_utils import get_input_files, InputFtrType, iterate_items_with_list_val


def _read_specified_features(inputs, feature_type2name):
    """Only reads in features specified in the DeText arguments"""
    required_inputs = {}
    for _, ftr_name_list in iterate_items_with_list_val(feature_type2name):
        for ftr_name in ftr_name_list:
            required_inputs[ftr_name] = inputs[ftr_name]
    return required_inputs


_FTR_TYPE_TO_SCHEMA = {
    InputFtrType.TARGET_COLUMN_NAME: tf.io.FixedLenFeature(shape=[], dtype=tf.string)
}


def _get_tfrecord_feature_parsing_schema(feature_type_2_name: dict):
    """Returns parsing schema for input TFRecord

    :param feature_type_2_name: Features mapping from feature types to feature names
    """
    ftr_name_2_schema = dict()
    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type_2_name):
        for ftr_name in ftr_name_lst:
            ftr_name_2_schema[ftr_name] = _FTR_TYPE_TO_SCHEMA[ftr_type]

    return ftr_name_2_schema


def _cast_features_to_smaller_dtype(example, feature_type_2_names: dict):
    """Casts tensor to smaller storage dtype. int64 -> int32, float64 -> float32"""

    def _cast_to_dtype_of_smaller_size(t):
        if t.dtype == tf.int64:
            return tf.cast(t, dtype=tf.int32)
        elif t.dtype == tf.float64:
            return tf.cast(t, dtype=tf.float32)
        else:
            return t

    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type_2_names):
        for ftr_name in ftr_name_lst:
            example[ftr_name] = _cast_to_dtype_of_smaller_size(example[ftr_name])
    return example


_FTR_TYPE_TO_DENSE_DEFAULT_VAL = {
    InputFtrType.TARGET_COLUMN_NAME: '',
}


def input_fn_tfrecord(input_pattern,
                      batch_size,
                      mode,
                      feature_type_2_name: dict,
                      block_length=100,
                      prefetch_size=tf.data.experimental.AUTOTUNE,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE,
                      input_pipeline_context=None):
    """
    Data input function for training given TFRecord
    """
    output_buffer_size = 1000

    input_files = get_input_files(input_pattern)
    feature_type_2_name = feature_type_2_name.copy()
    if len(input_files) > 1:  # Multiple input files
        # Preprocess files concurrently, and interleave blocks of block_length records from each file
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        # Shard input when using distributed training strategy
        if mode == tf.estimator.ModeKeys.TRAIN and input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
            dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                    input_pipeline_context.input_pipeline_id)

        dataset = dataset.shuffle(buffer_size=len(input_files))

        dataset = dataset.interleave(tf.data.TFRecordDataset, block_length=block_length,
                                     num_parallel_calls=num_parallel_calls)
    else:
        dataset = tf.data.TFRecordDataset(input_files[0])

    # Parse and preprocess data
    dataset = tfrecord_transform_fn(dataset,
                                    batch_size,
                                    mode,
                                    feature_type_2_name,
                                    output_buffer_size,
                                    prefetch_size)
    return dataset


def _split_features_and_labels(example, feature_type_2_name: dict):
    """Split inputs into two parts: features and label"""
    target_ftr_name = feature_type_2_name[InputFtrType.TARGET_COLUMN_NAME]
    labels = {
        target_ftr_name: example.pop(target_ftr_name)
    }

    return example, labels


def tfrecord_transform_fn(dataset,
                          batch_size,
                          mode,
                          feature_type_2_name,
                          output_buffer_size,
                          prefetch_size=tf.data.experimental.AUTOTUNE,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """ Preprocesses datasets including
        1. dataset shuffling
        2. record parsing
        3. padding and batching
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(output_buffer_size)
        dataset = dataset.repeat()

    def _process_data(record, features_schema):
        example = tf.io.parse_single_example(serialized=record, features=features_schema)
        example = _cast_features_to_smaller_dtype(example, feature_type_2_name)
        features, labels = _split_features_and_labels(example, feature_type_2_name)
        return features, labels

    features_schema = _get_tfrecord_feature_parsing_schema(feature_type_2_name)
    dataset = dataset.map(partial(_process_data, features_schema=features_schema),
                          num_parallel_calls=num_parallel_calls)

    dataset = (dataset
               .batch(batch_size, drop_remainder=True)
               .prefetch(prefetch_size))
    return dataset
