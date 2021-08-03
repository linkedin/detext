from functools import partial

import tensorflow as tf

from detext.train.constant import Constant
from detext.utils.parsing_utils import get_input_files, InputFtrType, TaskType, iterate_items_with_list_val, as_list


def _read_specified_features(inputs, feature_type2name):
    """Only reads in features specified in the DeText arguments"""
    required_inputs = {}
    for _, ftr_name_list in iterate_items_with_list_val(feature_type2name):
        for ftr_name in ftr_name_list:
            required_inputs[ftr_name] = inputs[ftr_name]
    return required_inputs


def _set_dense_ftrs_padding_shapes(nums_dense_ftrs: list):
    """Sets the dense feature padding shape"""
    if not nums_dense_ftrs:
        return

    Constant()._FTR_TYPE2PADDED_SHAPE[InputFtrType.DENSE_FTRS_COLUMN_NAMES] = []
    for n in nums_dense_ftrs:
        Constant()._FTR_TYPE2PADDED_SHAPE[InputFtrType.DENSE_FTRS_COLUMN_NAMES].append(tf.TensorShape([None, n]))


def _get_padded_shapes_and_values(feature_type2name: dict, nums_dense_ftrs: list):
    """Returns padded_shape and padd_values for each feature

    :param feature_type2name Map from feature types to feature names EXCLUDING 'label'
    """
    _set_dense_ftrs_padding_shapes(nums_dense_ftrs)

    ftr_name2padded_shapes = dict()
    ftr_name2padded_values = dict()

    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
        # Do not handle sparse features. It will be handled separately in the sparse batch function
        if ftr_type == InputFtrType.SPARSE_FTRS_COLUMN_NAMES:
            continue

        # Padded shapes and values already known and initialized
        if ftr_type in (InputFtrType.LABEL_COLUMN_NAME, InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME):
            continue

        # The last dimension of dense features is known and could be different from each other. Therefore, we put the
        #   padded shape for each dense features column separately
        if ftr_type == InputFtrType.DENSE_FTRS_COLUMN_NAMES:
            for ftr_name, padded_shape in zip(ftr_name_lst, Constant()._FTR_TYPE2PADDED_SHAPE[ftr_type]):
                ftr_name2padded_shapes[ftr_name] = padded_shape
                ftr_name2padded_values[ftr_name] = Constant()._FTR_TYPE2PADDED_VALUE[ftr_type]
            continue

        for ftr_name in ftr_name_lst:
            ftr_name2padded_shapes[ftr_name] = Constant()._FTR_TYPE2PADDED_SHAPE[ftr_type]
            ftr_name2padded_values[ftr_name] = Constant()._FTR_TYPE2PADDED_VALUE[ftr_type]

    label_padded_shapes = {
        feature_type2name[InputFtrType.LABEL_COLUMN_NAME]: tf.TensorShape([None]),
    }
    label_padded_values = {
        feature_type2name[InputFtrType.LABEL_COLUMN_NAME]: Constant()._LABEL_PADDING,
    }

    if InputFtrType.WEIGHT_COLUMN_NAME in feature_type2name:
        label_padded_shapes[feature_type2name[InputFtrType.WEIGHT_COLUMN_NAME]] = tf.TensorShape([])
        label_padded_values[feature_type2name[InputFtrType.WEIGHT_COLUMN_NAME]] = 1.0

    if InputFtrType.UID_COLUMN_NAME in feature_type2name:
        label_padded_shapes[feature_type2name[InputFtrType.UID_COLUMN_NAME]] = tf.TensorShape([])
        label_padded_values[feature_type2name[InputFtrType.UID_COLUMN_NAME]] = tf.cast(0, tf.int64)

    ftr_name2padded_shapes = (ftr_name2padded_shapes, label_padded_shapes)
    ftr_name2padded_values = (ftr_name2padded_values, label_padded_values)
    return ftr_name2padded_shapes, ftr_name2padded_values


def _sparse_ftrs_indices0(ftr_name):
    """Returns the name of the 0th axis indices for `ftr_name`"""
    return f"{ftr_name}_indices0"


def _sparse_ftrs_indices1(ftr_name):
    """Returns the name of the 1st axis indices for `ftr_name`"""
    return f"{ftr_name}_indices1"


def _sparse_ftrs_values(ftr_name):
    """Returns the name of the values for sparse feature `ftr_name`"""
    return f"{ftr_name}_values"


def add_ranking_sparse_feature_schema(ftr_name_2_schema, ftr_name):
    """Adds schema of ranking sparse feature `ftr_name` """
    ftr_name_2_schema[_sparse_ftrs_indices0(ftr_name)] = tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    ftr_name_2_schema[_sparse_ftrs_indices1(ftr_name)] = tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    ftr_name_2_schema[_sparse_ftrs_values(ftr_name)] = tf.io.VarLenFeature(dtype=tf.dtypes.float32)
    return ftr_name_2_schema


def add_classification_sparse_feature_schema(ftr_name_2_schema, ftr_name):
    """Adds schema of classification sparse feature `ftr_name` """
    ftr_name_2_schema[_sparse_ftrs_indices0(ftr_name)] = tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    ftr_name_2_schema[_sparse_ftrs_values(ftr_name)] = tf.io.VarLenFeature(dtype=tf.dtypes.float32)
    return ftr_name_2_schema


_TASK_TYPE_TO_ADD_SPARSE_FEATURE_SCHEMA_FN = {
    TaskType.RANKING: add_ranking_sparse_feature_schema,
    TaskType.CLASSIFICATION: add_classification_sparse_feature_schema,
}


def _get_tfrecord_feature_parsing_schema(feature_type2name: dict, ftr_type_to_schema, task_type):
    """Returns parsing schema for input TFRecord

    :param feature_type2name: Features mapping from feature types to feature names
    """
    add_sparse_feature_schema_fn = _TASK_TYPE_TO_ADD_SPARSE_FEATURE_SCHEMA_FN[task_type]

    ftr_name_2_schema = dict()
    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
        for ftr_name in ftr_name_lst:
            if ftr_type == InputFtrType.SPARSE_FTRS_COLUMN_NAMES:
                ftr_name_2_schema = add_sparse_feature_schema_fn(ftr_name_2_schema, ftr_name)
            else:
                ftr_name_2_schema[ftr_name] = ftr_type_to_schema[ftr_type]

    return ftr_name_2_schema


def _cast_features_to_smaller_dtype(example, feature_type2name: dict):
    """Casts tensor to smaller storage dtype. int64 -> int32, float64 -> float32"""

    def _cast_to_dtype_of_smaller_size(t):
        if t.dtype == tf.int64:
            return tf.cast(t, dtype=tf.int32)
        elif t.dtype == tf.float64:
            return tf.cast(t, dtype=tf.float32)
        else:
            return t

    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
        if ftr_type in {InputFtrType.TASK_ID_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME, InputFtrType.SPARSE_FTRS_COLUMN_NAMES}:
            continue
        for ftr_name in ftr_name_lst:
            example[ftr_name] = _cast_to_dtype_of_smaller_size(example[ftr_name])
    return example


def _ranking_sparse_ftrs_to_dense(example, ftr_name):
    """Converts ranking sparse features to dense format"""
    for sparse_ftr_name in [_sparse_ftrs_indices0(ftr_name), _sparse_ftrs_indices1(ftr_name),
                            _sparse_ftrs_values(ftr_name)]:
        example[sparse_ftr_name] = tf.sparse.to_dense(example[sparse_ftr_name])
    return example


def _classification_sparse_ftrs_to_dense(example, ftr_name):
    """Converts classification sparse features to dense format"""
    for sparse_ftr_name in [_sparse_ftrs_indices0(ftr_name), _sparse_ftrs_values(ftr_name)]:
        example[sparse_ftr_name] = tf.sparse.to_dense(example[sparse_ftr_name])
    return example


_TASK_TYPE_TO_SPARSE_FTRS_TO_DENSE_FN_DICT = {
    TaskType.RANKING: _ranking_sparse_ftrs_to_dense,
    TaskType.CLASSIFICATION: _classification_sparse_ftrs_to_dense
}


def _convert_ftrs_to_dense_tensor(example, feature_type2name, ftr_type_to_dense_default_val):
    """Converts the VarLenFeature to dense format"""
    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
        if ftr_type == InputFtrType.SPARSE_FTRS_COLUMN_NAMES:
            continue

        for ftr_name in ftr_name_lst:
            if ftr_type in ftr_type_to_dense_default_val:
                example[ftr_name] = tf.sparse.to_dense(example[ftr_name], default_value=ftr_type_to_dense_default_val[ftr_type])

    return example


def _convert_sparse_ftrs_indices_and_values_to_dense_tensor(example, feature_type2name, task_type):
    """Converts the VarLenFeature of sparse features to dense format"""
    task_type_to_sparse_ftrs_to_dense_fn = _TASK_TYPE_TO_SPARSE_FTRS_TO_DENSE_FN_DICT[task_type]
    if InputFtrType.SPARSE_FTRS_COLUMN_NAMES in feature_type2name:
        for ftr_name in as_list(feature_type2name[InputFtrType.SPARSE_FTRS_COLUMN_NAMES]):
            example = task_type_to_sparse_ftrs_to_dense_fn(example, ftr_name)

    return example


def _assemble_sparse_ftrs_ranking(example, feature_type2_name, nums_sparse_ftrs):
    """Assembles ranking sparse feature indices and values into tf.SparseFeature

    Indices and values are removed from example in this process. E.g.,
        input_example = {'sparse_ftrs_a_indices0': Tensor, 'sparse_ftrs_a_indices1': Tensor, 'sparse_ftrs_a_values': Tensor}
        output_example = {'sparse_ftrs_a': SparseTensor}
    """
    example = _convert_sparse_ftrs_indices_and_values_to_dense_tensor(example, feature_type2_name, TaskType.RANKING)
    ftr_name_lst = feature_type2_name.get(InputFtrType.SPARSE_FTRS_COLUMN_NAMES, [])
    labels = example[feature_type2_name[InputFtrType.LABEL_COLUMN_NAME]]
    list_size = tf.shape(labels)[-1]

    for ftr_name, num_sparse_ftrs in zip(ftr_name_lst, nums_sparse_ftrs):
        indices = tf.stack(
            [example.pop(_sparse_ftrs_indices0(ftr_name)), example.pop(_sparse_ftrs_indices1(ftr_name))],
            axis=1
        )
        example[ftr_name] = tf.SparseTensor(
            indices=indices,
            values=example.pop(_sparse_ftrs_values(ftr_name)),
            dense_shape=[list_size, num_sparse_ftrs]
        )
    return example


def _assemble_sparse_ftrs_classification(example, feature_type2_name, nums_sparse_ftrs):
    """Assembles classification sparse feature indices and values into tf.SparseFeature

    Indices and values are removed from example in this process. E.g.,
        input_example = {'sparse_ftrs_a_indices': Tensor, 'sparse_ftrs_a_values': Tensor}
        output_example = {'sparse_ftrs_a': SparseTensor}
    """
    example = _convert_sparse_ftrs_indices_and_values_to_dense_tensor(example, feature_type2_name, TaskType.CLASSIFICATION)
    ftr_name_lst = feature_type2_name.get(InputFtrType.SPARSE_FTRS_COLUMN_NAMES, [])

    for ftr_name, num_sparse_ftrs in zip(ftr_name_lst, nums_sparse_ftrs):
        indices = example.pop(_sparse_ftrs_indices0(ftr_name))
        indices = tf.expand_dims(indices, axis=-1)  # [N_non_zero, 1]
        values = example.pop(_sparse_ftrs_values(ftr_name))
        example[ftr_name] = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[num_sparse_ftrs]
        )
    return example


def _reshape_ftrs_to_group_wise(example, feature_type2name, nums_dense_ftrs: list):
    """Reshapes features from [num_fts] to [num_ftrs_each_doc, num_docs]

    :param example Mapping from feature types to feature names
    """
    num_docs = tf.size(input=example[feature_type2name[InputFtrType.LABEL_COLUMN_NAME]])

    for ftr_type in [InputFtrType.DENSE_FTRS_COLUMN_NAMES]:
        if ftr_type not in feature_type2name:
            continue

        # The last dimension of dense features is known and could be different from each other. Therefore, we explicitly specify
        #    the last dimension of dense features
        if ftr_type == InputFtrType.DENSE_FTRS_COLUMN_NAMES:
            for ftr_name, n in zip(as_list(feature_type2name[ftr_type]), nums_dense_ftrs):
                example[ftr_name] = tf.reshape(example[ftr_name], shape=[num_docs, n])
            continue

        for ftr_name in as_list(feature_type2name[ftr_type]):
            example[ftr_name] = tf.reshape(example[ftr_name], shape=[num_docs, -1])
    return example


def input_fn_tfrecord(input_pattern,
                      batch_size,
                      mode,
                      feature_type2name: dict,
                      nums_dense_ftrs: list,
                      nums_sparse_ftrs: list,
                      task_type=TaskType.RANKING,
                      block_length=100,
                      prefetch_size=tf.data.experimental.AUTOTUNE,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE,
                      input_pipeline_context=None):
    """
    Data input function for training given TFRecord
    """
    output_buffer_size = 1000

    input_files = get_input_files(input_pattern)
    feature_type2name = feature_type2name.copy()
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
    task_type_to_transform_fn = {
        # Ranking input transform function
        TaskType.RANKING:
            lambda dataset, batch_size, mode, feature_type2name, output_buffer_size, prefetch_size, nums_sparse_ftrs, nums_dense_ftrs:
            ranking_transform_fn(dataset=dataset,
                                 batch_size=batch_size,
                                 mode=mode,
                                 feature_type2name=feature_type2name,
                                 nums_dense_ftrs=nums_dense_ftrs,
                                 nums_sparse_ftrs=nums_sparse_ftrs,
                                 output_buffer_size=output_buffer_size,
                                 prefetch_size=prefetch_size),
        # Classification input transform function
        TaskType.CLASSIFICATION:
            lambda dataset, batch_size, mode, feature_type2name, output_buffer_size, prefetch_size, nums_sparse_ftrs, *args:
            classification_transform_fn(dataset=dataset,
                                        batch_size=batch_size,
                                        mode=mode,
                                        feature_type2name=feature_type2name,
                                        output_buffer_size=output_buffer_size,
                                        prefetch_size=prefetch_size,
                                        nums_sparse_ftrs=nums_sparse_ftrs
                                        ),
        # Binary classification input transform function
        TaskType.BINARY_CLASSIFICATION:
            lambda dataset, batch_size, mode, feature_type2name, output_buffer_size, prefetch_size, *args:
            classification_transform_fn(dataset=dataset,
                                        batch_size=batch_size,
                                        mode=mode,
                                        feature_type2name=feature_type2name,
                                        output_buffer_size=output_buffer_size,
                                        prefetch_size=prefetch_size,
                                        nums_sparse_ftrs=nums_sparse_ftrs
                                        ),
    }
    return task_type_to_transform_fn[task_type](dataset,
                                                batch_size,
                                                mode,
                                                feature_type2name,
                                                output_buffer_size,
                                                prefetch_size,
                                                nums_sparse_ftrs,
                                                nums_dense_ftrs
                                                )


def _squeeze_ftrs(example, feature_type2name, feature_type_to_squeeze):
    """Squeezes features from 1d to scalar"""
    for ftr_type in feature_type_to_squeeze:
        ftr_name_lst = feature_type2name.get(ftr_type, [])
        ftr_name_lst = as_list(ftr_name_lst)
        for ftr_name in ftr_name_lst:
            example[ftr_name] = tf.squeeze(example[ftr_name], axis=0)
    return example


def _split_features_and_labels(example, feature_type2name: dict):
    """Split inputs into two parts: features and label"""
    label_ftr_name = feature_type2name[InputFtrType.LABEL_COLUMN_NAME]
    labels = {label_ftr_name: tf.cast(example.pop(label_ftr_name), dtype=tf.float32)}

    for input_feature_type in (InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME):
        if input_feature_type in feature_type2name:
            ftr_name = feature_type2name[input_feature_type]
            labels[ftr_name] = example.pop(ftr_name)
    return example, labels


def _add_default_ftr_field(features, labels, feature_type2name: dict):
    """ Adds default feature fields if not exist"""

    # Default weight as feature. Set to 1.0 if not present in data
    if InputFtrType.WEIGHT_COLUMN_NAME not in feature_type2name:
        labels.setdefault(Constant()._DEFAULT_WEIGHT_FTR_NAME, tf.ones(tf.shape(labels[feature_type2name[InputFtrType.LABEL_COLUMN_NAME]])[0],
                                                                       dtype=tf.float32))

    # Default uid as feature for detext integration, will be -1 by default if not present in data
    if InputFtrType.UID_COLUMN_NAME not in feature_type2name:
        labels.setdefault(Constant()._DEFAULT_UID_FTR_NAME, -tf.ones(tf.shape(labels[feature_type2name[InputFtrType.LABEL_COLUMN_NAME]])[0],
                                                                     dtype=tf.int64))

    return features, labels


def ranking_transform_fn(dataset,
                         batch_size,
                         mode,
                         feature_type2name,
                         nums_dense_ftrs,
                         nums_sparse_ftrs,
                         output_buffer_size,
                         prefetch_size=tf.data.experimental.AUTOTUNE,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """ Preprocesses datasets for ranking task including
        1. dataset shuffling
        2. record parsing
        3. padding and batching
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(output_buffer_size)
        dataset = dataset.repeat()

    def _process_data(record, features_schema, nums_dense_ftrs):
        example = tf.io.parse_single_example(serialized=record, features=features_schema)

        example = _cast_features_to_smaller_dtype(example, feature_type2name)
        example = _convert_ftrs_to_dense_tensor(example, feature_type2name, Constant()._RANKING_FTR_TYPE_TO_DENSE_DEFAULT_VAL)
        example = _assemble_sparse_ftrs_ranking(example, feature_type2name, nums_sparse_ftrs)

        feature_type_to_squeeze = [InputFtrType.QUERY_COLUMN_NAME, InputFtrType.USER_ID_COLUMN_NAMES,
                                   InputFtrType.USER_TEXT_COLUMN_NAMES, InputFtrType.WEIGHT_COLUMN_NAME,
                                   InputFtrType.TASK_ID_COLUMN_NAME,
                                   InputFtrType.UID_COLUMN_NAME]
        example = _squeeze_ftrs(example, feature_type2name, feature_type_to_squeeze)
        example = _reshape_ftrs_to_group_wise(example, feature_type2name, nums_dense_ftrs)

        example = _read_specified_features(example, feature_type2name)
        features, labels = _split_features_and_labels(example, feature_type2name)
        return features, labels

    features_schema = _get_tfrecord_feature_parsing_schema(feature_type2name, Constant()._RANKING_FTR_TYPE_TO_SCHEMA, TaskType.RANKING)
    dataset = dataset.map(partial(_process_data, features_schema=features_schema, nums_dense_ftrs=nums_dense_ftrs),
                          num_parallel_calls=num_parallel_calls)

    dataset = batch_dataset(dataset, feature_type2name, nums_dense_ftrs, batch_size).map(
        partial(_add_default_ftr_field, feature_type2name=feature_type2name),
        num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.prefetch(prefetch_size)
    return dataset


def _sparse_batch_fn(features, labels, feature_type2name, padded_shapes, padded_values, batch_size):
    """Batches features and labels where features is a dictionary that contains sparse features(tf.SparseTensor) """
    inputs_padded_shapes, labels_padded_shapes = padded_shapes
    inputs_padded_values, labels_padded_values = padded_values

    sparse_inputs = {}
    for sparse_ftrs_name in feature_type2name[InputFtrType.SPARSE_FTRS_COLUMN_NAMES]:
        sparse_inputs[sparse_ftrs_name] = features.pop(sparse_ftrs_name).batch(batch_size)

    features = {ftr_name: ds.padded_batch(batch_size=batch_size,
                                          padded_shapes=inputs_padded_shapes[ftr_name],
                                          padding_values=inputs_padded_values[ftr_name]) for ftr_name, ds in features.items()}
    features.update(sparse_inputs)

    labels = {ftr_name: ds.padded_batch(batch_size=batch_size,
                                        padded_shapes=labels_padded_shapes[ftr_name],
                                        padding_values=labels_padded_values[ftr_name]) for ftr_name, ds in labels.items()}
    return tf.data.Dataset.zip((features, labels))


def batch_dataset(dataset: tf.data.Dataset, feature_type2name, nums_dense_ftrs, batch_size):
    """Performs batching on ranking dataset

    When there's no sparse features, padded_batch() is enough. When there are sparse features, we use batching function specific for sparse features
    """
    padded_shapes, padded_values = _get_padded_shapes_and_values(feature_type2name, nums_dense_ftrs)

    # Use padded_batch() if no sparse features
    if InputFtrType.SPARSE_FTRS_COLUMN_NAMES not in feature_type2name:
        # drop_remainder=True to avoid input batch_size=0 issue in evaluation mode in multi gpu training
        return dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padded_values, drop_remainder=True)

    sparse_batch_fn = partial(_sparse_batch_fn, feature_type2name=feature_type2name,
                              padded_shapes=padded_shapes,
                              padded_values=padded_values,
                              batch_size=batch_size)
    # drop_remainder=True to avoid input batch_size=0 issue in evaluation mode in multi gpu training
    return dataset.window(batch_size, drop_remainder=True).flat_map(sparse_batch_fn)


def classification_transform_fn(dataset,
                                batch_size,
                                mode,
                                feature_type2name,
                                output_buffer_size,
                                nums_sparse_ftrs,
                                prefetch_size=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """ Preprocesses datasets for classification task including
        1. dataset shuffling
        2. record parsing
        3. padding and batching
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(output_buffer_size)
        dataset = dataset.repeat()

    def _process_data(record, features_schema):
        example = tf.io.parse_single_example(serialized=record, features=features_schema)

        example = _cast_features_to_smaller_dtype(example, feature_type2name)
        example = _convert_ftrs_to_dense_tensor(example, feature_type2name, Constant()._CLASSIFICATION_FTR_TYPE_TO_DENSE_DEFAULT_VAL)
        example = _assemble_sparse_ftrs_classification(example, feature_type2name, nums_sparse_ftrs)

        feature_type_to_squeeze = [InputFtrType.QUERY_COLUMN_NAME, InputFtrType.USER_ID_COLUMN_NAMES,
                                   InputFtrType.USER_TEXT_COLUMN_NAMES, InputFtrType.DOC_TEXT_COLUMN_NAMES, InputFtrType.DOC_ID_COLUMN_NAMES,
                                   InputFtrType.WEIGHT_COLUMN_NAME,
                                   InputFtrType.TASK_ID_COLUMN_NAME,
                                   InputFtrType.UID_COLUMN_NAME]
        example = _squeeze_ftrs(example, feature_type2name, feature_type_to_squeeze)
        example = _read_specified_features(example, feature_type2name)
        features, labels = _split_features_and_labels(example, feature_type2name)
        return features, labels

    features_schema = _get_tfrecord_feature_parsing_schema(feature_type2name, Constant()._CLASSIFICATION_FTR_TYPE_TO_SCHEMA, TaskType.CLASSIFICATION)
    dataset = dataset.map(partial(_process_data, features_schema=features_schema),
                          num_parallel_calls=num_parallel_calls)

    # drop_remainder=True to avoid input batch_size=0 issue in evaluation mode in multi gpu training
    dataset = dataset.batch(batch_size, drop_remainder=True).map(
        partial(_add_default_ftr_field, feature_type2name=feature_type2name),
        num_parallel_calls=num_parallel_calls
    ).prefetch(prefetch_size)
    return dataset
