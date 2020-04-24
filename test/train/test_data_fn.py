import numpy as np
import os
import tensorflow as tf

from detext.train import data_fn
from detext.utils import vocab_utils


class TestData(tf.test.TestCase):
    """Unit test for data_fn."""
    PAD_ID = 3
    CLS_ID = 101

    def testInputFnBuilderTfrecord(self):
        """Test function input_fn_builder() in eval mode"""
        res_dir = os.path.dirname(__file__) + '/../resources'

        # create a vocab table
        vocab_table = vocab_utils.read_tf_vocab(res_dir + '/vocab.txt', '[UNK]')

        # dataset dir
        data_dir = os.path.join(res_dir, 'train', 'dataset', 'tfrecord')

        # create a dataset.
        # Read schema
        # Parse and process data in dataset
        feature_names = (
            'label', 'query', 'doc_completedQuery', 'usr_headline', 'usr_skills', 'usr_currTitles', 'usrId_currTitles',
            'docId_completedQuery', 'wide_ftrs', 'weight')

        batch_size = 2
        dataset = data_fn.input_fn(input_pattern=data_dir,
                                   metadata_path=None,
                                   batch_size=batch_size,
                                   mode=tf.estimator.ModeKeys.EVAL,
                                   vocab_table=vocab_table,
                                   vocab_table_for_id_ftr=vocab_table,
                                   feature_names=feature_names,
                                   CLS='[CLS]',
                                   SEP='[SEP]',
                                   PAD='[PAD]',
                                   PAD_FOR_ID_FTR='[PAD]',
                                   max_len=16,
                                   cnn_filter_window_size=1)

        # Make iterator
        iterator = dataset.make_initializable_iterator()
        batch_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sess.run([iterator.initializer])
            batch_data_val, = sess.run([batch_data])
            features, label = batch_data_val

            # First dimension of data should be batch_size
            for ftr_name in feature_names:
                if ftr_name != 'label':
                    self.assertTrue(ftr_name in features)
                    self.assertTrue(features[ftr_name].shape[0] == batch_size)

            self.assertTrue(label['label'].shape[0] == batch_size)

            doc_completedQuery = features['doc_completedQuery']
            docId_completedQuery = features['docId_completedQuery']
            usr_currTitles = features['usr_currTitles']
            usrId_currTitles = features['usrId_currTitles']

            # vocab[PAD] == PAD_ID
            self.assertTrue(doc_completedQuery[0, 0, -1] == self.PAD_ID)
            self.assertTrue(docId_completedQuery[0, 0, -1] == self.PAD_ID)

            # vocab[CLS] == CLS_ID
            self.assertTrue(np.all(doc_completedQuery[0, 0, 0] == self.CLS_ID))
            self.assertTrue(np.all(usr_currTitles[0, 0] == self.CLS_ID))

            # No CLS in id feature
            self.assertTrue(np.all(docId_completedQuery[:, :, 0] != self.CLS_ID))

            # In this TFRecord file, we populate docId_completeQuery using doc_completedQuery
            # doc id feature should be the same as doc text feature except CLS and SEP addition
            # Here we make sure this is correct for the first sample
            for text_arr, id_arr in zip(doc_completedQuery[0], docId_completedQuery[0]):
                self.assertAllEqual(text_arr[text_arr != self.PAD_ID][1:-1], id_arr[id_arr != self.PAD_ID])

            # In this TFRecord file, we populate usrId_currTitles using usr_currTitles
            # usr id feature should be the same as usr text feature except CLS and SEP addition
            for text_arr, id_arr in zip(usr_currTitles, usrId_currTitles):
                self.assertAllEqual(text_arr[text_arr != self.PAD_ID][1:-1], id_arr[id_arr != self.PAD_ID])


if __name__ == "__main__":
    tf.test.main()
