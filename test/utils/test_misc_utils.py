import os
import tensorflow as tf
import pickle

from detext.utils import misc_utils

root_dir = os.path.join(os.path.dirname(__file__), "../resources")


class TestMiscUtils(tf.test.TestCase):
    def testLoadFtrMeanStd(self):
        """Unit test for _load_ftr_mean_std """
        filepath_spark = os.path.join(root_dir, "tmp_ftr_mean_std.fromspark")

        true_mean = [1, 2, 3, 4]
        true_std = [3, 2, 1, 3]

        # Test pickle file
        separator = ","
        with open(filepath_spark, 'w') as fout:
            fout.write("# Feature mean std file. Mean is first line, std is second line\n")
            fout.write(separator.join([str(x) for x in true_mean]) + "\n")
            fout.write(separator.join([str(x) for x in true_std]) + "\n")

        ftr_mean, ftr_std = misc_utils._load_ftr_mean_std(filepath_spark)
        self.assertEquals(ftr_mean, true_mean)
        self.assertEquals(ftr_std, true_std)
        os.remove(filepath_spark)

        # Test pickle file
        filepath_pkl = os.path.join(root_dir, "tmp_ftr_mean_std.pkl")
        with tf.gfile.Open(filepath_pkl, 'wb') as fout:
            pickle.dump((true_mean, true_std), fout, protocol=2)
        ftr_mean, ftr_std = misc_utils._load_ftr_mean_std(filepath_pkl)
        self.assertEquals(ftr_mean, true_mean)
        self.assertEquals(ftr_std, true_std)
        os.remove(filepath_pkl)
