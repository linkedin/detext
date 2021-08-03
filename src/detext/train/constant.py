import tensorflow as tf
import tensorflow_ranking as tfr

from detext.utils.parsing_utils import InputFtrType
from detext.metaclass import SingletonMeta


class Constant(metaclass=SingletonMeta):
    def __init__(self):
        self._LABEL_PADDING = tfr.data._PADDING_LABEL
        self._DEFAULT_WEIGHT_FTR_NAME = InputFtrType.WEIGHT_COLUMN_NAME
        self._DEFAULT_UID_FTR_NAME = InputFtrType.UID_COLUMN_NAME

        self._FTR_TYPE2PADDED_SHAPE = {
            InputFtrType.QUERY_COLUMN_NAME: tf.TensorShape([]),
            InputFtrType.USER_TEXT_COLUMN_NAMES: tf.TensorShape([]),
            InputFtrType.DOC_TEXT_COLUMN_NAMES: tf.TensorShape([None]),
            InputFtrType.USER_ID_COLUMN_NAMES: tf.TensorShape([]),
            InputFtrType.DOC_ID_COLUMN_NAMES: tf.TensorShape([None]),
            InputFtrType.TASK_ID_COLUMN_NAME: tf.TensorShape([]),
        }

        self._FTR_TYPE2PADDED_VALUE = {
            InputFtrType.QUERY_COLUMN_NAME: '',
            InputFtrType.USER_TEXT_COLUMN_NAMES: '',
            InputFtrType.DOC_TEXT_COLUMN_NAMES: '',
            InputFtrType.USER_ID_COLUMN_NAMES: '',
            InputFtrType.DOC_ID_COLUMN_NAMES: '',
            InputFtrType.TASK_ID_COLUMN_NAME: tf.cast(0, tf.int64),
            InputFtrType.DENSE_FTRS_COLUMN_NAMES: 0.0,
        }

        self._RANKING_FTR_TYPE_TO_DENSE_DEFAULT_VAL = {
            InputFtrType.QUERY_COLUMN_NAME: '',
            InputFtrType.USER_TEXT_COLUMN_NAMES: '',
            InputFtrType.USER_ID_COLUMN_NAMES: '',
            InputFtrType.DOC_TEXT_COLUMN_NAMES: '',
            InputFtrType.DOC_ID_COLUMN_NAMES: '',
            InputFtrType.DENSE_FTRS_COLUMN_NAMES: 0.0,
            InputFtrType.LABEL_COLUMN_NAME: self._LABEL_PADDING,
            InputFtrType.SPARSE_FTRS_COLUMN_NAMES: 0
        }

        self._CLASSIFICATION_FTR_TYPE_TO_DENSE_DEFAULT_VAL = {
            InputFtrType.DENSE_FTRS_COLUMN_NAMES: 0.0,
            InputFtrType.SPARSE_FTRS_COLUMN_NAMES: 0
        }

        self._RANKING_FTR_TYPE_TO_SCHEMA = {
            InputFtrType.WEIGHT_COLUMN_NAME: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
            InputFtrType.UID_COLUMN_NAME: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
            InputFtrType.TASK_ID_COLUMN_NAME: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),

            InputFtrType.LABEL_COLUMN_NAME: tf.io.VarLenFeature(dtype=tf.float32),

            InputFtrType.QUERY_COLUMN_NAME: tf.io.VarLenFeature(dtype=tf.string),
            InputFtrType.USER_TEXT_COLUMN_NAMES: tf.io.VarLenFeature(dtype=tf.string),
            InputFtrType.USER_ID_COLUMN_NAMES: tf.io.VarLenFeature(dtype=tf.string),

            InputFtrType.DOC_TEXT_COLUMN_NAMES: tf.io.VarLenFeature(dtype=tf.string),
            InputFtrType.DOC_ID_COLUMN_NAMES: tf.io.VarLenFeature(dtype=tf.string),

            InputFtrType.DENSE_FTRS_COLUMN_NAMES: tf.io.VarLenFeature(dtype=tf.float32),
        }

        self._CLASSIFICATION_FTR_TYPE_TO_SCHEMA = {
            InputFtrType.WEIGHT_COLUMN_NAME: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
            InputFtrType.UID_COLUMN_NAME: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
            InputFtrType.TASK_ID_COLUMN_NAME: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),

            InputFtrType.LABEL_COLUMN_NAME: tf.io.FixedLenFeature(shape=[], dtype=tf.float32),

            InputFtrType.QUERY_COLUMN_NAME: tf.io.FixedLenFeature(shape=[1], dtype=tf.string),  # not used
            InputFtrType.USER_TEXT_COLUMN_NAMES: tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
            InputFtrType.USER_ID_COLUMN_NAMES: tf.io.FixedLenFeature(shape=[1], dtype=tf.string),

            InputFtrType.DOC_TEXT_COLUMN_NAMES: tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
            InputFtrType.DOC_ID_COLUMN_NAMES: tf.io.FixedLenFeature(shape=[1], dtype=tf.string),

            InputFtrType.DENSE_FTRS_COLUMN_NAMES: tf.io.VarLenFeature(dtype=tf.float32),
        }
