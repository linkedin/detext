""" Parsing and IO utilities"""
import codecs
import json
import os
from collections import Mapping

import six
import tensorflow as tf
from absl import logging

_MAX_FILE = 50
_RANDOM_SEED = 4321

_HPARAM_FILE = 'hparams'


class InputFtrType:
    """ Input feature types

    These are the feature types directly read from user inputs
    """
    TARGET_COLUMN_NAME = 'target_column_name'


class OutputFtrType:
    """Output feature types

    These are the feature types that will be provided in the model output
    """
    PREDICTED_SCORES = 'predicted_scores'
    PREDICTED_TEXTS = 'predicted_texts'
    EXIST_PREFIX = 'exist_prefix'


class InternalFtrType:
    """Internal feature types

    These are the feature types that will be passed to layers inside the model
    """
    LAST_MEMORY_STATE = 'last_memory_state'
    LAST_CARRY_STATE = 'last_carry_state'

    SENTENCES = 'sentences'
    MIN_LEN = 'min_len'
    MAX_LEN = 'max_len'

    NUM_CLS = 'num_cls'
    NUM_SEP = 'num_sep'

    TOKENIZED_IDS = 'tokenized_ids'
    TOKENIZED_TEXTS = 'tokenized_texts'
    LENGTH = 'length'
    EMBEDDED = 'embedded'

    LOGIT = 'logits'
    LABEL = 'labels'
    RNN_OUTPUT = '_rnn_output'
    SAMPLE_ID = '_sample_id'

    EXIST_KEY = 'exist_key'
    EXIST_PREFIX = OutputFtrType.EXIST_PREFIX
    COMPLETION_INDICES = 'completion_indices'
    COMPLETION_VOCAB_MASK = 'completion_vocab_mask'

    SEQUENCE_TO_ENCODE = 'sequence_to_encode'


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


class HParams(Mapping):
    """
    Hyper parameter class that behaves similar to the original tf 1.x HParams class w.r.t. functionality used in Smart Compose
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
