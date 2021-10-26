import os
import shutil

import tensorflow as tf

from . import data_setup


class TestCase(tf.test.TestCase, data_setup.DataSetup):
    """ Unit test class"""

    def assertDictAllEqual(self, a: dict, b: dict):
        """ Checks that two dictionaries are the same """
        self.assertIsInstance(a, dict)
        self.assertIsInstance(b, dict)
        self.assertAllEqual(a.keys(), b.keys())

        for k in a.keys():
            self.assertAllEqual(a[k], b[k])

    def _cleanUp(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)
