""" Parsing and related IO utilities"""
import codecs
import json
import os
import pickle
import random
from collections import Mapping

import numpy as np
import six
import tensorflow as tf
from absl import logging

_MAX_FILE = 50
_RANDOM_SEED = 4321

_HPARAM_FILE = 'hparams'


class InputFtrType:
    """ Feature types """
    QUERY_COLUMN_NAME = 'query_column_name'
    WEIGHT_COLUMN_NAME = 'weight_column_name'
    TASK_ID_COLUMN_NAME = 'task_id_column_name'
    UID_COLUMN_NAME = 'uid_column_name'
    LABEL_COLUMN_NAME = 'label_column_name'

    USER_TEXT_COLUMN_NAMES = 'user_text_column_names'
    DOC_TEXT_COLUMN_NAMES = 'doc_text_column_names'
    USER_ID_COLUMN_NAMES = 'user_id_column_names'
    DOC_ID_COLUMN_NAMES = 'doc_id_column_names'
    DENSE_FTRS_COLUMN_NAMES = 'dense_ftrs_column_names'
    SPARSE_FTRS_COLUMN_NAMES = 'sparse_ftrs_column_names'

    SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES = 'shallow_tower_sparse_ftrs_column_names'


class OutputFtrType:
    """Output feature types """
    DETEXT_CLS_PROBABILITIES = 'detext_cls_probabilities'
    DETEXT_CLS_LOGITS = 'detext_cls_logits'
    DETEXT_CLS_PREDICTED_LABEL = 'detext_cls_predicted_label'

    DETEXT_RANKING_SCORES = 'detext_ranking_scores'


class InternalFtrType:
    """ Feature types produced during network feed-forward pass in the DeText model"""
    QUERY_FTRS = 'query_ftrs'
    DOC_FTRS = 'doc_ftrs'
    USER_FTRS = 'user_ftrs'

    WIDE_FTRS = 'wide_ftrs'
    DENSE_FTRS = 'dense_ftrs'
    SPARSE_FTRS = "sparse_ftrs"
    INTERACTION_FTRS = 'interaction_ftrs'

    SEQ_OUTPUTS = 'seq_outputs'
    LAST_MEMORY_STATE = 'last_memory_state'
    LAST_CARRY_STATE = 'last_carry_state'

    SENTENCES = 'sentences'
    MIN_LEN = 'min_len'
    MAX_LEN = 'max_len'
    NUM_CLS = 'num_cls'
    NUM_SEP = 'num_sep'
    TOKENIZED_IDS = 'tokenized_ids'
    LENGTH = 'length'

    EMBEDDED = "embedded"

    FTRS_TO_SCORE = "ftrs_to_score"

    DEEP_FTR_BAG = 'deep_ftr_bag'
    WIDE_FTR_BAG = 'wide_ftr_bag'
    SHALLOW_TOWER_FTR_BAG = 'shallow_tower_ftr_bag'
    MULTITASK_FTR_BAG = 'multitask_ftr_bag'


class TaskType:
    """ Task types """
    RANKING = 'ranking'
    # General classification mode for num_classes >= 2. Model output is a one-dim tensor (e.g., one prob for each class)
    CLASSIFICATION = 'classification'
    # Multiclass classification mode. Same as classification mode. Multiclass classification usually means num_classes >= 3 according to
    #   [Wikipedia](https://en.wikipedia.org/wiki/Multiclass_classification). Model output is a one-dim tensor (e.g., one prob for each class)
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    # Binary classification mode. Model output is a scalar (e.g., prob for sample being positive)
    BINARY_CLASSIFICATION = 'binary_classification'


def as_list(value):
    """Returns value as a list

    If the value is not a list, it will be converted as a list containing one single item
    """
    if isinstance(value, list):
        return value
    return [value]


def iterate_items_with_list_val(dct):
    """Helper function that iterates the dict items

    If the value is not a list, it will be converted as a list containing one single item
    """
    return [(key, as_list(value)) for key, value in dct.items()]


def get_feature_types():
    """ Returns the list of feature names defined in class FtrName"""
    constant_to_name_tuples = filter(lambda x: not x[0].startswith(('_', '__')), vars(InputFtrType).items())  # [(QUERY, query), (WEIGHT, weight), ...]
    feature_types = [t[1] for t in constant_to_name_tuples]
    return feature_types


def get_feature_nums(feature_name2num, feature_type2name, feature_type):
    """ Returns the list of feature numbers wrt given feature type"""
    return [feature_name2num[ftr_name] for ftr_name in feature_type2name.get(feature_type, [])]


class HParams(Mapping):
    """
    Hyper parameter class that behaves similar to the original tf 1.x HParams class w.r.t. functionality used in DeText
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        for k in self.__dict__:
            yield k

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __repr__(self):
        return self.__dict__.__repr__()

    def to_json(self):
        """Serializes hparams to json

        Reference: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/hparam.py#L529
        """

        def remove_callables(x):
            """Omit callable elements from input with arbitrary nesting."""
            if isinstance(x, dict):
                return {k: remove_callables(v) for k, v in six.iteritems(x)
                        if not callable(v)}
            elif isinstance(x, list):
                return [remove_callables(i) for i in x if not callable(i)]
            return x

        return json.dumps(remove_callables(self.__dict__))


def _get_hparam_path(out_dir):
    return os.path.join(out_dir, _HPARAM_FILE)


def save_hparams(out_dir, hparams):
    """Saves hparams"""
    hparams_file = _get_hparam_path(out_dir)
    logging.info("Saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.compat.v1.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())


def load_hparams(out_dir):
    """Loads hparams"""
    hparams_file = _get_hparam_path(out_dir)
    if tf.compat.v1.gfile.Exists(hparams_file):
        logging.info("Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(tf.compat.v1.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = HParams(**hparams_values)
            except ValueError:
                logging.error("Can't load hparams file")
                return None
        return hparams
    else:
        return None


def get_input_files(input_patterns):
    """Returns a list of file paths that match every pattern in input_patterns

    :param input_patterns a comma-separated string
    :return list of file paths
    """
    input_files = []
    for input_pattern in input_patterns.split(","):
        if tf.io.gfile.isdir(input_pattern):
            input_pattern = os.path.join(input_pattern, '*')
        input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))
    return input_files


def load_ftr_mean_std(path):
    """ Loads mean and standard deviation from given file """
    with tf.compat.v1.gfile.Open(path, 'rb') as fin:
        if path.endswith("fromspark"):
            data = fin.readlines()
            # Line 0 is printing message, line 1 is feature mean, line 2 is feature std
            ftr_mean = [float(x.strip()) for x in data[1].decode("utf-8").split(',')]
            ftr_std = [float(x.strip()) for x in data[2].decode("utf-8").split(',')]
        else:
            ftr_mean, ftr_std = pickle.load(fin)
    # Put std val 0 -> 1 to avoid zero division error
    for i in range(len(ftr_std)):
        if ftr_std[i] == 0:
            ftr_std[i] = 1
    return ftr_mean, ftr_std


def get_num_fields(prefix, feature_names):
    """Returns the number of feature names that starts with prefix"""
    return sum([name.startswith(prefix) for name in feature_names])


def estimate_steps_per_epoch(input_pattern, batch_size):
    """ Estimates train steps per epoch for tfrecord files

    Counting exact total number of examples is time consuming and unnecessary,
    We count the first file and use the total file size to estimate total number of examples.
    """
    input_files = get_input_files(input_pattern)

    file_1st = input_files[0]
    file_1st_num_examples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(file_1st))
    logging.info("number of examples in first file: {0}".format(file_1st_num_examples))

    file_1st_size = tf.compat.v1.gfile.GFile(file_1st).size()
    logging.info("first file size: {0}".format(file_1st_size))

    file_size_num_example_ratio = float(file_1st_size) / file_1st_num_examples

    estimated_num_examples = sum([int(tf.compat.v1.gfile.GFile(fn).size() / file_size_num_example_ratio)
                                  for fn in input_files])
    logging.info("Estimated number of examples: {0}".format(estimated_num_examples))

    return int(estimated_num_examples / batch_size)


def compute_mean_std(input_files_patten, output_file, num_dense_ftrs):
    """Computes mean and std for dense_ftrs from given tfrecord/avro

    This function is kept for historical reason. For new users, please DO NOT use it.
    """
    random.seed(_RANDOM_SEED)
    file_format = input_files_patten.split('.')[-1]
    all_ftrs = [[] for _ in range(num_dense_ftrs)]

    # Read data, apply normalization on dense features
    files = tf.io.gfile.glob(input_files_patten)
    random.shuffle(files)
    if file_format in ('tfrecords', 'tfrecord'):
        for input_file in files[:_MAX_FILE]:
            logging.info(input_file)
            for example in tf.compat.v1.python_io.tf_record_iterator(input_file):
                result = tf.train.Example.FromString(example)
                dense_ftrs = result.features.feature['wide_ftrs'].float_list.value
                for i, f in enumerate(dense_ftrs):
                    i2 = i % num_dense_ftrs
                    all_ftrs[i2].append(f)
    else:
        raise ValueError("Unsupported file format %s." % file_format)

    mean_value = [0] * num_dense_ftrs
    std_value = [0] * num_dense_ftrs
    for i in range(num_dense_ftrs):
        logging.info('%d %f %f' % (i, np.mean(all_ftrs[i]), np.std(all_ftrs[i])))
        mean_value[i] = np.mean(all_ftrs[i]).item()
        std_value[i] = np.std(all_ftrs[i]).item()
    with tf.io.gfile.GFile(output_file, 'wb') as fout:
        pickle.dump((mean_value, std_value), fout, protocol=2)

    return mean_value, std_value
