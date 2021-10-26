from os.path import join as path_join

import tensorflow as tf
from official.utils.misc import distribution_utils

from smart_compose.train import data_fn
from smart_compose.utils import vocab_utils
from smart_compose.utils.parsing_utils import InputFtrType, iterate_items_with_list_val
from smart_compose.utils.testing.test_case import TestCase


class TestData(TestCase):
    """Unit test for data_fn.py"""
    _, vocab_tf_table = vocab_utils.read_tf_vocab(TestCase.vocab_file, '[UNK]')
    vocab_table = TestCase.vocab_table_py

    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'

    PAD_ID = vocab_table[PAD]
    SEP_ID = vocab_table[SEP]
    CLS_ID = vocab_table[CLS]

    target_column_name = 'query'

    def testInputFnBuilderTfrecord(self):
        """ Tests function input_fn_builder() """
        one_device_strategy = distribution_utils.get_distribution_strategy('one_device', num_gpus=0)
        for strategy in [None, one_device_strategy]:
            self._testInputFnBuilderTfrecord(strategy)

    def _testInputFnBuilderTfrecord(self, strategy):
        """ Tests function input_fn_builder() for given strategy """
        data_dir = path_join(self.data_dir)

        # Create a dataset
        # Read schema
        # Parse and process data in dataset
        feature_type_2_name = {
            InputFtrType.TARGET_COLUMN_NAME: self.target_column_name,
        }

        def _input_fn_tfrecord(ctx):
            return data_fn.input_fn_tfrecord(input_pattern=data_dir,
                                             batch_size=batch_size,
                                             mode=tf.estimator.ModeKeys.EVAL,
                                             feature_type_2_name=feature_type_2_name,
                                             input_pipeline_context=ctx)

        batch_size = 2
        if strategy is not None:
            dataset = strategy.experimental_distribute_datasets_from_function(_input_fn_tfrecord)
        else:
            dataset = _input_fn_tfrecord(None)

        # Make iterator
        for features, label in dataset:
            for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type_2_name):
                if ftr_type in (InputFtrType.TARGET_COLUMN_NAME,):
                    self.assertLen(ftr_name_lst, 1), f'Length for current ftr type ({ftr_type}) should be 1'
                    ftr_name = ftr_name_lst[0]
                    self.assertIn(ftr_name, label)
                    continue

            # Check source and target text shape
            self.assertAllEqual(label[self.target_column_name].shape, [batch_size])

            break


if __name__ == "__main__":
    tf.test.main()
