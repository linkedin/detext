import os
import tensorflow as tf

from detext.train import data_fn
from detext.utils import vocab_utils


class TestData(tf.test.TestCase):
    """Unit test for data_fn."""
    def testInputFnBuilder(self):
        # resource directory
        res_dir = os.path.dirname(__file__) + '/../resources'
        # create a vocab table
        vocab_table = vocab_utils.read_tf_vocab(res_dir + '/vocab.txt', '[UNK]')
        # dataset dir
        input_file_pattern = os.path.join(res_dir, 'sample_data', '*.tfrecord')

        # apply input_fn_builder()
        feature_names = ('query', 'label', 'wide_ftrs', 'doc_titles')
        dataset = data_fn.input_fn(
            input_pattern=input_file_pattern,
            batch_size=2,
            mode=tf.estimator.ModeKeys.EVAL,
            vocab_table=vocab_table,
            feature_names=feature_names,
            CLS="[CLS]", SEP="[SEP]", PAD="[PAD]",
            max_len=5,
            cnn_filter_window_size=1
        )
        iterator = dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sess.run([iterator.initializer])
            batch_data_val, = sess.run([batch_data])
            features, label = batch_data_val
            self.assertAllEqual([10, 6], features['group_size'])
            self.assertAllEqual([[101, 2129, 2079, 2017, 3972],
                                 [101, 2129, 2000, 9167, 2026]], features['query'])


if __name__ == "__main__":
    tf.test.main()
