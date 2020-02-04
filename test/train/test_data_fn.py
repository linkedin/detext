import os
import tensorflow as tf

from detext.train import data_fn
from detext.utils import vocab_utils


class TestData(tf.test.TestCase):
    """Unit test for data_fn."""

    # def testInputFnBuilderAvro(self):
    #     """Test function input_fn_builder() in eval mode"""
    #     res_dir = os.path.dirname(__file__) + '/../resources'
    #
    #     # create a vocab table
    #     vocab_table = vocab_utils.read_tf_vocab(res_dir + '/vocab.txt', '[UNK]')
    #
    #     # dataset dir
    #     input_file_pattern = os.path.join(res_dir, 'sample_data', '*.tfrecords')
    #
    #     # create a dataset.
    #     # Read schema
    #     # Parse and process data in dataset
    #     feature_names = ('query', 'label', 'wide_ftrs',
    #                      'doc_job_title', 'doc_job_company')
    #     batch_size = 2
    #     dataset = data_fn.input_fn(input_pattern=input_file_pattern,
    #                                batch_size=batch_size,
    #                                mode=tf.estimator.ModeKeys.EVAL,
    #                                vocab_table=vocab_table,
    #                                feature_names=feature_names,
    #                                CLS='[CLS]',
    #                                SEP='[SEP]',
    #                                PAD='[PAD]',
    #                                max_len=16,
    #                                cnn_filter_window_size=1)
    #
    #     # Make iterator
    #     iterator = dataset.make_initializable_iterator()
    #     batch_data = iterator.get_next()
    #
    #     with tf.Session() as sess:
    #         sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #         sess.run([iterator.initializer])
    #         batch_data_val, = sess.run([batch_data])
    #         features, label = batch_data_val
    #
    #         # First dimension of data should be batch_size
    #         for ftr_name in feature_names:
    #             if ftr_name != 'label':
    #                 self.assertTrue(ftr_name in features)
    #                 self.assertTrue(features[ftr_name].shape[0] == batch_size)
    #         self.assertTrue('label' in label)
    #         self.assertTrue(label['label'].shape[0] == batch_size)
    #
    #         self.assertTrue(features['doc_job_title'][0][0][-1] == 3)  # vocab[PAD] == 3
    #         # First token of text field should be [CLS]
    #         for ftr_name in feature_names:
    #             if ftr_name.startswith('doc_'):
    #                 for sample in features[ftr_name]:
    #                     for doc_data in sample:
    #                         if doc_data[0] != 101:
    #                             print(doc_data)
    #                         self.assertTrue(doc_data[0] == 101)  # vocab[CLS] == 101

    def testInputFnBuilder(self):
        # resource directory
        res_dir = os.path.dirname(__file__) + '/../resources'
        # create a vocab table
        vocab_table = vocab_utils.read_tf_vocab(res_dir + '/vocab.txt', '[UNK]')
        # dataset dir
        input_file_pattern = os.path.join(res_dir, 'sample_data', '*.tfrecord')

        # apply input_fn_builder()
        feature_names = ('query', 'label', 'wide_ftrs', 'doc_titles', 'doc_examples')
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
