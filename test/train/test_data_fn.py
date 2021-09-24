from itertools import product

import tensorflow as tf
from detext.train import data_fn, constant
from detext.utils import vocab_utils
from detext.utils.parsing_utils import InputFtrType, TaskType, iterate_items_with_list_val
from detext.utils.testing.data_setup import DataSetup
from official.utils.misc import distribution_utils


class TestData(tf.test.TestCase, DataSetup):
    """Unit test for data_fn."""
    _, vocab_tf_table = vocab_utils.read_tf_vocab(DataSetup.vocab_file, '[UNK]')
    vocab_table = vocab_utils.read_vocab(DataSetup.vocab_file)

    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'

    PAD_ID = vocab_table[PAD]
    SEP_ID = vocab_table[SEP]
    CLS_ID = vocab_table[CLS]

    nums_sparse_ftrs = [20]

    def testRankingInputFnBuilderTfrecord(self):
        """ Tests function input_fn_builder() """
        one_device_strategy = distribution_utils.get_distribution_strategy('one_device', num_gpus=0)
        feature_type2name_list = [
            # Contains sparse features
            {InputFtrType.LABEL_COLUMN_NAME: 'label',
             InputFtrType.QUERY_COLUMN_NAME: 'query',
             InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_headline', 'doc_title'],
             InputFtrType.USER_TEXT_COLUMN_NAMES: ['user_headline', 'user_title'],
             InputFtrType.DOC_ID_COLUMN_NAMES: ['doc_headline_id'],
             InputFtrType.USER_ID_COLUMN_NAMES: ['user_headline_id'],
             InputFtrType.DENSE_FTRS_COLUMN_NAMES: ['dense_ftrs'],
             InputFtrType.SPARSE_FTRS_COLUMN_NAMES: ['sparse_ftrs'],
             InputFtrType.WEIGHT_COLUMN_NAME: 'weight'
             },
            # No sparse features
            {InputFtrType.LABEL_COLUMN_NAME: 'label',
             InputFtrType.QUERY_COLUMN_NAME: 'query',
             InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_headline', 'doc_title'],
             InputFtrType.USER_TEXT_COLUMN_NAMES: ['user_headline', 'user_title'],
             InputFtrType.DOC_ID_COLUMN_NAMES: ['doc_headline_id'],
             InputFtrType.USER_ID_COLUMN_NAMES: ['user_headline_id'],
             InputFtrType.DENSE_FTRS_COLUMN_NAMES: ['dense_ftrs'],
             InputFtrType.WEIGHT_COLUMN_NAME: 'weight'
             },
            # Sparse features only
            {InputFtrType.LABEL_COLUMN_NAME: 'label',
             InputFtrType.SPARSE_FTRS_COLUMN_NAMES: ['sparse_ftrs']}
        ]
        strategy_list = [None, one_device_strategy]

        for strategy, feature_type2name in product(strategy_list, feature_type2name_list):
            self._testRankingInputFnBuilderTfrecord(strategy, feature_type2name)

    def _testRankingInputFnBuilderTfrecord(self, strategy, feature_type2name):
        """ Tests function input_fn_builder() for given strategy """
        data_dir = self.ranking_data_dir
        feature_name2num = {'dense_ftrs': 2, 'sparse_ftrs': self.nums_sparse_ftrs[0]}

        def _input_fn_tfrecord(ctx):
            return data_fn.input_fn_tfrecord(input_pattern=data_dir,
                                             batch_size=batch_size,
                                             mode=tf.estimator.ModeKeys.EVAL,
                                             feature_type2name=feature_type2name,
                                             feature_name2num=feature_name2num,
                                             input_pipeline_context=ctx)

        batch_size = 2
        if strategy is not None:
            dataset = strategy.distribute_datasets_from_function(_input_fn_tfrecord)
        else:
            dataset = _input_fn_tfrecord(None)

        # Make iterator
        for features, label in dataset:
            for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
                if ftr_type in (InputFtrType.LABEL_COLUMN_NAME, InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME):
                    self.assertLen(ftr_name_lst, 1), f'Length for current ftr type ({ftr_type}) should be 1'
                    ftr_name = ftr_name_lst[0]
                    self.assertIn(ftr_name, label)
                    continue

                for ftr_name in ftr_name_lst:
                    self.assertIn(ftr_name, features)
                    # First dimension of data should be batch_size
                    self.assertTrue(features[ftr_name].shape[0] == batch_size)

            weight_ftr_name = feature_type2name.get(InputFtrType.WEIGHT_COLUMN_NAME, constant.Constant()._DEFAULT_WEIGHT_FTR_NAME)
            self.assertAllEqual(tf.shape(label[weight_ftr_name]), [batch_size])

            uid_ftr_name = feature_type2name.get(InputFtrType.UID_COLUMN_NAME, constant.Constant()._DEFAULT_UID_FTR_NAME)
            self.assertAllEqual(tf.shape(label[uid_ftr_name]), [batch_size])

            # First dimension of data should be batch_size
            self.assertEqual(label['label'].shape[0], batch_size)

            if InputFtrType.DOC_TEXT_COLUMN_NAMES in feature_type2name:
                self.assertAllEqual(features['doc_title'],
                                    tf.constant(
                                        [["document title 1", b"title 2 ?", b"doc title 3 ?", b"doc title 4 ?"],
                                         ["document title 1", b"title 2 ?", b"doc title 3 ?", b""]]
                                    ))

            if InputFtrType.DOC_ID_COLUMN_NAMES in feature_type2name:
                self.assertAllEqual(features['doc_headline_id'],
                                    tf.constant(
                                        [[b"document headline id 1", b"headline id 2 ?", b"doc headline id 3 ?", b"doc headline id 4 ?"],
                                         [b"document headline id 1", b"headline id 2 ?", b"doc headline id 3 ?", b""]]
                                    ))

            if InputFtrType.USER_TEXT_COLUMN_NAMES in feature_type2name:
                self.assertAllEqual(features['user_title'],
                                    tf.constant(
                                        [b"user title", b"user title"]
                                    ))
            if InputFtrType.USER_ID_COLUMN_NAMES in feature_type2name:
                self.assertAllEqual(features['user_headline_id'],
                                    tf.constant(
                                        [b"user headline id", b"user headline id"]
                                    ))

            if InputFtrType.DENSE_FTRS_COLUMN_NAMES in feature_type2name:
                self.assertAllEqual(features['dense_ftrs'],
                                    tf.constant(
                                        [[[23.0, 14.0], [44.0, -1.0], [22.0, 19.0], [22.0, 19.0]],
                                         [[23.0, 14.0], [44.0, -1.0], [22.0, 19.0], [0.0, 0.0]]]
                                    ))

            if InputFtrType.SPARSE_FTRS_COLUMN_NAMES in feature_type2name:
                self.assertAllEqual(tf.sparse.to_dense(features['sparse_ftrs']),
                                    tf.sparse.to_dense(tf.SparseTensor(indices=[[0, 0, 1],
                                                                                [0, 0, 5],
                                                                                [0, 1, 0],
                                                                                [0, 2, 2],
                                                                                [0, 3, 8],
                                                                                [1, 0, 1],
                                                                                [1, 0, 5],
                                                                                [1, 1, 0],
                                                                                [1, 2, 2]],
                                                                       values=[1., 5., 7., 12., -8., 1., 5., 7., 12.],
                                                                       dense_shape=[batch_size, 4, self.nums_sparse_ftrs[0]]))
                                    )

            # Only check the first batch
            break

    def testClassificationInputFnBuilderTfrecord(self):
        """Test classification input reader in eval mode"""
        data_dir = self.cls_data_dir

        feature_type2name = {
            InputFtrType.LABEL_COLUMN_NAME: 'label',
            InputFtrType.DOC_TEXT_COLUMN_NAMES: ['query_text'],
            InputFtrType.USER_TEXT_COLUMN_NAMES: ['user_headline'],
            InputFtrType.DENSE_FTRS_COLUMN_NAMES: 'dense_ftrs',
        }
        feature_name2num = {
            'dense_ftrs': 8
        }

        batch_size = 2
        dataset = data_fn.input_fn_tfrecord(input_pattern=data_dir,
                                            batch_size=batch_size,
                                            mode=tf.estimator.ModeKeys.EVAL,
                                            task_type=TaskType.CLASSIFICATION,
                                            feature_type2name=feature_type2name,
                                            feature_name2num=feature_name2num)

        for features, label in dataset:
            # First dimension of data should be batch_size
            for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
                if ftr_type in (InputFtrType.LABEL_COLUMN_NAME, InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME):
                    self.assertLen(ftr_name_lst, 1), f'Length for current ftr type ({ftr_type}) should be 1'
                    ftr_name = ftr_name_lst[0]
                    self.assertIn(ftr_name, label)
                    continue
                for ftr_name in ftr_name_lst:
                    self.assertIn(ftr_name, features)
                    self.assertEqual(features[ftr_name].shape[0], batch_size)

            weight_ftr_name = feature_type2name.get(InputFtrType.WEIGHT_COLUMN_NAME, constant.Constant()._DEFAULT_WEIGHT_FTR_NAME)
            self.assertAllEqual(tf.shape(label[weight_ftr_name]), [batch_size])

            uid_ftr_name = feature_type2name.get(InputFtrType.UID_COLUMN_NAME, constant.Constant()._DEFAULT_UID_FTR_NAME)
            self.assertAllEqual(tf.shape(label[uid_ftr_name]), [batch_size])

            self.assertAllEqual(label['label'].shape, [batch_size])

    def testBinaryClassificationInputFnBuilderTfrecord(self):
        """Test binary classification input reader """
        data_dir = self.binary_cls_data_dir

        feature_type2name = {
            InputFtrType.LABEL_COLUMN_NAME: 'label',
            InputFtrType.SPARSE_FTRS_COLUMN_NAMES: ['sparse_ftrs'],
            InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES: ['shallow_tower_sparse_ftrs', 'sparse_ftrs']
        }
        feature_name2num = {
            'sparse_ftrs': 20,
            'shallow_tower_sparse_ftrs': 20
        }

        batch_size = 2
        dataset = data_fn.input_fn_tfrecord(input_pattern=data_dir,
                                            batch_size=batch_size,
                                            mode=tf.estimator.ModeKeys.EVAL,
                                            task_type=TaskType.BINARY_CLASSIFICATION,
                                            feature_type2name=feature_type2name,
                                            feature_name2num=feature_name2num
                                            )

        for features, label in dataset:
            # First dimension of data should be batch_size
            for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
                if ftr_type in (InputFtrType.LABEL_COLUMN_NAME, InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME):
                    self.assertLen(ftr_name_lst, 1), f'Length for current ftr type ({ftr_type}) should be 1'
                    ftr_name = ftr_name_lst[0]
                    self.assertIn(ftr_name, label)
                    continue
                for ftr_name in ftr_name_lst:
                    self.assertIn(ftr_name, features)
                    self.assertEqual(features[ftr_name].shape[0], batch_size)

            weight_ftr_name = feature_type2name.get(InputFtrType.WEIGHT_COLUMN_NAME, constant.Constant()._DEFAULT_WEIGHT_FTR_NAME)
            self.assertAllEqual(tf.shape(label[weight_ftr_name]), [batch_size])

            uid_ftr_name = feature_type2name.get(InputFtrType.UID_COLUMN_NAME, constant.Constant()._DEFAULT_UID_FTR_NAME)
            self.assertAllEqual(tf.shape(label[uid_ftr_name]), [batch_size])

            self.assertAllEqual(label['label'].shape, [batch_size])
            self.assertAllEqual(tf.sparse.to_dense(features['sparse_ftrs']),
                                tf.sparse.to_dense(
                                    tf.SparseTensor(indices=[[0, 0],
                                                             [0, 2],
                                                             [0, 7],
                                                             [1, 0],
                                                             [1, 2],
                                                             [1, 7]],
                                                    values=[1, 0, 7, 1, 0, 7],
                                                    dense_shape=[batch_size, self.nums_sparse_ftrs[0]])
                                )
                                )

            # Only check first batch
            break

    def testRankingMultitaskInputFnBuilderTfrecord(self):
        """Test additional input from multitask training in eval mode"""
        data_dir = self.ranking_data_dir

        # Test minimum features required for multitask jobs
        feature_type2name = {
            InputFtrType.LABEL_COLUMN_NAME: 'label',
            InputFtrType.QUERY_COLUMN_NAME: 'query',
            InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_headline', 'doc_title'],
            InputFtrType.USER_TEXT_COLUMN_NAMES: ['user_headline', 'user_title'],
            InputFtrType.DOC_ID_COLUMN_NAMES: ['doc_headline_id'],
            InputFtrType.USER_ID_COLUMN_NAMES: ['user_headline_id'],
            InputFtrType.DENSE_FTRS_COLUMN_NAMES: ['dense_ftrs'],
            InputFtrType.WEIGHT_COLUMN_NAME: 'weight',
            InputFtrType.TASK_ID_COLUMN_NAME: 'task_id_field'
        }
        feature_name2num = {
            'dense_ftrs': 2
        }

        batch_size = 5
        dataset = data_fn.input_fn_tfrecord(input_pattern=data_dir,
                                            batch_size=batch_size,
                                            mode=tf.estimator.ModeKeys.EVAL,
                                            feature_type2name=feature_type2name,
                                            feature_name2num=feature_name2num)

        for features, label in dataset:
            # First dimension of data should be batch_size
            for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
                if ftr_type in (InputFtrType.LABEL_COLUMN_NAME, InputFtrType.WEIGHT_COLUMN_NAME, InputFtrType.UID_COLUMN_NAME):
                    self.assertLen(ftr_name_lst, 1), f'Length for current ftr type ({ftr_type}) should be 1'
                    ftr_name = ftr_name_lst[0]
                    self.assertIn(ftr_name, label)
                    continue
                for ftr_name in ftr_name_lst:
                    self.assertIn(ftr_name, features)
                    self.assertEqual(features[ftr_name].shape[0], batch_size)

            weight_ftr_name = feature_type2name.get(InputFtrType.WEIGHT_COLUMN_NAME, constant.Constant()._DEFAULT_WEIGHT_FTR_NAME)
            self.assertAllEqual(tf.shape(label[weight_ftr_name]), [batch_size])

            uid_ftr_name = feature_type2name.get(InputFtrType.UID_COLUMN_NAME, constant.Constant()._DEFAULT_UID_FTR_NAME)
            self.assertAllEqual(tf.shape(label[uid_ftr_name]), [batch_size])

            # First dimension of data should be batch_size
            self.assertEqual(label['label'].shape[0], batch_size)

            task_ids = features['task_id_field']

            # Check task_id dimension size
            self.assertEqual(len(task_ids.shape), 1)

            # Check task_id value in the sample data
            for t_id in task_ids:
                self.assertAllEqual(t_id, 5)


if __name__ == "__main__":
    tf.test.main()
