import os

import tensorflow as tf

from smart_compose.utils import parsing_utils
from smart_compose.utils.parsing_utils import InputFtrType
from smart_compose.utils.testing.test_case import TestCase


class TestParsingUtils(TestCase):
    """Unit test for parsing_utils.py"""
    atol = 1e-3

    def testIterateItemsWithListVal(self):
        """Tests iterate_items_with_list_val"""
        dct_lst = [{'a': 'a'},
                   {'a': ['a']}]
        expected_result_lst = [[('a', ['a'])],
                               [('a', ['a'])]]
        assert len(dct_lst) == len(expected_result_lst), 'Number of test data and result must match'

        for dct, expected_result in zip(dct_lst, expected_result_lst):
            self.assertCountEqual(expected_result, list(parsing_utils.iterate_items_with_list_val(dct)))

    def testGetFeatureTypes(self):
        """Tests get_feature_types() """
        self.assertCountEqual(
            [InputFtrType.TARGET_COLUMN_NAME],
            parsing_utils.get_feature_types())

    def testHparamsLoadAndSave(self):
        """Tests loading and saving of hparams"""
        hparams = parsing_utils.HParams(a=1, b=2, c=[1, 2, 3])
        parsing_utils.save_hparams(self.resource_dir, hparams)
        loaded_hparams = parsing_utils.load_hparams(self.resource_dir)
        self.assertEqual(hparams, loaded_hparams)
        os.remove(parsing_utils._get_hparam_path(self.resource_dir))

    def testEstimateStepsPerEpoch(self):
        """Tests estimate_steps_per_epoch() """
        num_record = parsing_utils.estimate_steps_per_epoch(self.data_dir, 1)
        self.assertEqual(num_record, 40)


if __name__ == '__main__':
    tf.test.main()
