import os
import pickle

import tensorflow as tf
from detext.utils import parsing_utils
from detext.utils.parsing_utils import InputFtrType

root_dir = os.path.join(os.path.dirname(__file__), "../resources")
test_tfrecord_path = os.path.join(root_dir, 'train', 'dataset', 'tfrecord', 'test.tfrecord')


class TestParsingUtils(tf.test.TestCase):
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

    def testGetFeatureNames(self):
        """Tests get_feature_names() """
        self.assertCountEqual(
            [InputFtrType.QUERY_COLUMN_NAME, InputFtrType.LABEL_COLUMN_NAME, InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.TASK_ID_COLUMN_NAME,
             InputFtrType.UID_COLUMN_NAME, InputFtrType.DOC_TEXT_COLUMN_NAMES,
             InputFtrType.USER_TEXT_COLUMN_NAMES, InputFtrType.DOC_ID_COLUMN_NAMES, InputFtrType.USER_ID_COLUMN_NAMES,
             InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES,
             InputFtrType.DENSE_FTRS_COLUMN_NAMES, InputFtrType.SPARSE_FTRS_COLUMN_NAMES],
            parsing_utils.get_feature_types())

    def testHparamsLoadAndSave(self):
        """Tests loading and saving of hparams"""
        hparams = parsing_utils.HParams(a=1, b=2, c=[1, 2, 3])
        parsing_utils.save_hparams(root_dir, hparams)
        loaded_hparams = parsing_utils.load_hparams(root_dir)
        self.assertEqual(hparams, loaded_hparams)
        os.remove(parsing_utils._get_hparam_path(root_dir))

    def testComputeFtrMeanStd(self):
        """Tests compute_ftr_mean_std() """
        true_mean = [3310.9523809523807, 0.05952380952380952, 5.440476190476191]
        true_std = [7346.840887180146, 0.23660246326609274, 3.1028066805800445]

        output_file = os.path.join(root_dir, 'tmp_ftr_mean_std_output')
        mean, std = parsing_utils.compute_mean_std(test_tfrecord_path, output_file, 3)
        os.remove(output_file)

        self.assertAllClose(mean, true_mean, atol=self.atol)
        self.assertAllClose(std, true_std, atol=self.atol)

    def testLoadFtrMeanStd(self):
        """Tests load_ftr_mean_std() """
        true_mean = [1, 2, 3, 4]
        true_std = [3, 2, 1, 3]

        # Test file generated from spark
        separator = ","
        filepath_spark = os.path.join(root_dir, "tmp_ftr_mean_std.fromspark")
        with open(filepath_spark, 'w') as fout:
            fout.write("# Feature mean std file. Mean is first line, std is second line\n")
            fout.write(separator.join([str(x) for x in true_mean]) + "\n")
            fout.write(separator.join([str(x) for x in true_std]) + "\n")

        ftr_mean, ftr_std = parsing_utils.load_ftr_mean_std(filepath_spark)
        self.assertEqual(ftr_mean, true_mean)
        self.assertEqual(ftr_std, true_std)
        os.remove(filepath_spark)

        # Test pickle file
        filepath_pkl = os.path.join(root_dir, "tmp_ftr_mean_std.pkl")
        with tf.compat.v1.gfile.Open(filepath_pkl, 'wb') as fout:
            pickle.dump((true_mean, true_std), fout, protocol=2)
        ftr_mean, ftr_std = parsing_utils.load_ftr_mean_std(filepath_pkl)
        self.assertEqual(ftr_mean, true_mean)
        self.assertEqual(ftr_std, true_std)
        os.remove(filepath_pkl)

    def testEstimateStepsPerEpoch(self):
        """Tests estimate_steps_per_epoch() """
        num_record = parsing_utils.estimate_steps_per_epoch(test_tfrecord_path, 1)
        self.assertEqual(num_record, 10)

    def testGetNumFields(self):
        """Tests get_num_fields() """
        num_fields = parsing_utils.get_num_fields('doc_', ['doc_headline', 'docId_title'])
        self.assertEqual(num_fields, 1)


if __name__ == '__main__':
    tf.test.main()
