import tensorflow as tf

from detext.layers import feature_grouper
from detext.layers.feature_grouper import FeatureGrouper
from detext.utils import vocab_utils
from detext.utils.parsing_utils import InputFtrType
from detext.utils.testing.data_setup import DataSetup


class TestFeatureGrouper(tf.test.TestCase, DataSetup):
    """Unit test for feature_grouper.py"""
    _, vocab_tf_table = vocab_utils.read_tf_vocab(DataSetup.vocab_file, '[UNK]')
    vocab_table = vocab_utils.read_vocab(DataSetup.vocab_file)

    PAD_ID = vocab_table[DataSetup.PAD]
    SEP_ID = vocab_table[DataSetup.SEP]
    CLS_ID = vocab_table[DataSetup.CLS]
    UNK_ID = vocab_table[DataSetup.UNK]

    max_filter_window_size = 0

    def testFeatureGrouperKerasInput(self):
        """Tests FeatureGrouper with tf.keras.Input"""
        nums_dense_ftrs = [2, 3]
        nums_sparse_ftrs = [10, 30]
        layer = FeatureGrouper()
        inputs = {
            InputFtrType.QUERY_COLUMN_NAME: tf.keras.Input(shape=(), dtype='string'),
            InputFtrType.USER_TEXT_COLUMN_NAMES: [tf.keras.Input(shape=(), dtype='string')],
            InputFtrType.USER_ID_COLUMN_NAMES: [tf.keras.Input(shape=(), dtype='string')],
            InputFtrType.DOC_TEXT_COLUMN_NAMES: [tf.keras.Input(shape=(None,), dtype='string')],
            InputFtrType.DOC_ID_COLUMN_NAMES: [tf.keras.Input(shape=(None,), dtype='string')],
            InputFtrType.DENSE_FTRS_COLUMN_NAMES: [tf.keras.Input(shape=(num_dense_ftrs,), dtype='float32') for num_dense_ftrs in nums_dense_ftrs],
            InputFtrType.SPARSE_FTRS_COLUMN_NAMES: [tf.keras.Input(shape=(num_sparse_ftrs,), dtype='float32', sparse=True)
                                                    for num_sparse_ftrs in nums_sparse_ftrs]
        }
        outputs = layer(inputs)
        self.assertLen(outputs, len(inputs))

    def testFeatureGrouperTensor(self):
        """Tests FeatureGrouper with tensor input"""
        layer = FeatureGrouper()
        inputs = {InputFtrType.QUERY_COLUMN_NAME: tf.constant(['batch 1 user 1 build',
                                                               'batch 2 user 2 word'], dtype=tf.string),
                  InputFtrType.DENSE_FTRS_COLUMN_NAMES: [tf.constant([[1, 1], [2, 2]], dtype=tf.float32),
                                                         tf.constant([[0], [1]], dtype=tf.float32)],
                  InputFtrType.SPARSE_FTRS_COLUMN_NAMES: [tf.sparse.from_dense(tf.constant([[1, 0], [2, 0]], dtype=tf.float32)),
                                                          tf.sparse.from_dense(tf.constant([[1], [1]], dtype=tf.float32))]
                  }
        expected_result = {InputFtrType.QUERY_COLUMN_NAME: tf.constant(['batch 1 user 1 build',
                                                                        'batch 2 user 2 word'], dtype=tf.string),
                           InputFtrType.DENSE_FTRS_COLUMN_NAMES: tf.constant([[1, 1, 0],
                                                                              [2, 2, 1]]),
                           InputFtrType.SPARSE_FTRS_COLUMN_NAMES: [tf.constant([[1, 0],
                                                                                [2, 0]], dtype=tf.float32),
                                                                   tf.constant([[1], [1]], dtype=tf.float32)]
                           }
        outputs = layer(inputs)

        self.assertEqual(len(outputs), len(expected_result)), "Outputs must have the same shape"
        for ftr_type, expected_ftr in expected_result.items():
            output = outputs[ftr_type]
            if ftr_type == InputFtrType.SPARSE_FTRS_COLUMN_NAMES:
                output = [tf.sparse.to_dense(t) for t in output]
                for e, o in zip(expected_ftr, output):
                    self.assertAllEqual(e, o)
                continue
            self.assertAllEqual(expected_ftr, output)

    def testConcatFtrOnLastDim(self):
        """Tests concatenate features on last dimension"""
        tensor_lst = [tf.constant([1, 2, 3], dtype='int32'), tf.constant([4, 5, 6], dtype='int32')]
        result = feature_grouper.concat_on_last_axis_dense(tensor_lst)
        expected_output = tf.constant([1, 2, 3, 4, 5, 6], dtype='int32')
        self.assertAllEqual(result, expected_output)


if __name__ == '__main__':
    tf.test.main()
