import tensorflow as tf


class TestTF2(tf.test.TestCase):
    """Test whether TF2 is set up successfully"""

    def test_eager_execution(self):
        self.assertTrue(tf.executing_eagerly())
        self.assertTrue(tf.__version__.startswith('2.'))


if __name__ == "__main__":
    tf.test.main()
