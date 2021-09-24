import copy
import os
import shutil
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from detext.train import model
from detext.utils.parsing_utils import HParams, InputFtrType, TaskType
from detext.utils.testing.data_setup import DataSetup


class TestModel(tf.test.TestCase, DataSetup):
    """Tests model.py"""
    task_id = tf.constant([1, 0])
    nums_dense_ftrs = [5]

    # Dense features and labels
    ranking_dense_ftrs = tf.random.uniform(shape=[2, 3, nums_dense_ftrs[0]])
    cls_dense_ftrs = tf.random.uniform(shape=[2, nums_dense_ftrs[0]])
    labels = tf.constant([[1, 0, 1], [0, 0, 1]])

    # Config and hparams
    text_encoder_param = DataSetup.cnn_param
    id_encoder_param = DataSetup.id_encoder_param
    rep_layer_param = DataSetup.rep_layer_param

    QUERY_COLUMN_NAME_VAL = 'query'
    DENSE_FTRS_COLUMN_NAME_VAL = 'dense_ftrs'
    SPARSE_FTRS_COLUMN_NAME_VAL = 'sparse_ftrs'
    ranking_inputs = OrderedDict(sorted({QUERY_COLUMN_NAME_VAL: DataSetup.query,
                                         'usr_headline': DataSetup.user_fields[0],
                                         'usr_title': DataSetup.user_fields[1],
                                         'usrId_headline': DataSetup.user_id_fields[0],
                                         'usrId_title': DataSetup.user_id_fields[1],

                                         'doc_headline': DataSetup.ranking_doc_fields[0],
                                         'doc_title': DataSetup.ranking_doc_fields[1],
                                         'docId_headline': DataSetup.ranking_doc_id_fields[0],
                                         'docId_title': DataSetup.ranking_doc_id_fields[1],
                                         DENSE_FTRS_COLUMN_NAME_VAL: ranking_dense_ftrs,
                                         SPARSE_FTRS_COLUMN_NAME_VAL: DataSetup.ranking_sparse_features[0],
                                         }.items()))
    cls_inputs = OrderedDict(sorted({QUERY_COLUMN_NAME_VAL: DataSetup.query,
                                     'usr_headline': DataSetup.user_fields[0],
                                     'usr_title': DataSetup.user_fields[1],
                                     'usrId_headline': DataSetup.user_id_fields[0],
                                     'usrId_title': DataSetup.user_id_fields[1],

                                     'doc_headline': DataSetup.cls_doc_fields[0],
                                     'doc_title': DataSetup.cls_doc_fields[1],
                                     'docId_headline': DataSetup.cls_doc_id_fields[0],
                                     'docId_title': DataSetup.cls_doc_id_fields[1],
                                     DENSE_FTRS_COLUMN_NAME_VAL: cls_dense_ftrs,
                                     SPARSE_FTRS_COLUMN_NAME_VAL: DataSetup.cls_sparse_features[0],
                                     }.items()))
    feature_type2name = {InputFtrType.QUERY_COLUMN_NAME: QUERY_COLUMN_NAME_VAL,
                         InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_headline', 'doc_title'],
                         InputFtrType.DOC_ID_COLUMN_NAMES: ['docId_headline', 'docId_title'],
                         InputFtrType.USER_TEXT_COLUMN_NAMES: ['usr_headline', 'usr_title'],
                         InputFtrType.USER_ID_COLUMN_NAMES: ['usrId_headline', 'usrId_title'],
                         InputFtrType.DENSE_FTRS_COLUMN_NAMES: [DENSE_FTRS_COLUMN_NAME_VAL],
                         InputFtrType.SPARSE_FTRS_COLUMN_NAMES: [SPARSE_FTRS_COLUMN_NAME_VAL]}
    feature_name2num = {DENSE_FTRS_COLUMN_NAME_VAL: nums_dense_ftrs[0],
                        SPARSE_FTRS_COLUMN_NAME_VAL: DataSetup.nums_sparse_ftrs[0]}

    # deep match param
    deep_match_param = HParams(feature_name2num=feature_name2num,
                               use_dense_ftrs=True,
                               use_deep=True,
                               use_sparse_ftrs=True,
                               sparse_embedding_size=DataSetup.sparse_embedding_size,
                               sparse_embedding_cross_ftr_combiner='concat',
                               sparse_embedding_same_ftr_combiner='sum',
                               has_query=True,
                               emb_sim_func=['inner'],
                               rep_layer_param=rep_layer_param,
                               ftr_mean=None, ftr_std=None,
                               num_hidden=[10],
                               rescale_dense_ftrs=False,
                               num_classes=1,
                               task_ids=None)

    def testModelNetworkRanking(self):
        """Tests whether model can produce correct ranking output"""
        detext_model = model.create_detext_model(self.feature_type2name, task_type=TaskType.RANKING, **self.deep_match_param)
        outputs = detext_model.generate_training_scores(self.ranking_inputs)
        self.assertAllEqual(tf.shape(outputs), [2, 3])

    def testModelNetworkClassification(self):
        """Tests whether model can produce correct classification output"""
        num_classes = 3
        deep_match_param = copy.copy(self.deep_match_param)
        deep_match_param.num_classes = num_classes

        detext_model = model.create_detext_model(self.feature_type2name, task_type=TaskType.CLASSIFICATION, **deep_match_param)
        outputs = detext_model.generate_training_scores(self.cls_inputs)
        self.assertAllEqual(tf.shape(outputs), [2, num_classes])

    def testModelNetworkBinaryClassification(self):
        """Tests whether model can produce correct binary classification output"""
        num_classes = 1
        deep_match_param = copy.copy(self.deep_match_param)
        deep_match_param.num_classes = num_classes

        detext_model = model.create_detext_model(self.feature_type2name, task_type=TaskType.BINARY_CLASSIFICATION, **deep_match_param)
        outputs = detext_model.generate_training_scores(self.cls_inputs)
        self.assertAllEqual(tf.shape(outputs), [2])

    def testModelNetworkMultitaskRanking(self):
        """Tests whether model can produce correct ranking output in multitask training"""
        deep_match_param = copy.copy(self.deep_match_param)
        deep_match_param.task_ids = [0, 1]

        task_id_column_name_val = 'task_id'
        feature_type2name = copy.copy(self.feature_type2name)
        feature_type2name[InputFtrType.TASK_ID_COLUMN_NAME] = task_id_column_name_val
        inputs = copy.copy(self.ranking_inputs)
        inputs[task_id_column_name_val] = tf.constant([1, 0], dtype=tf.dtypes.int64)

        detext_model = model.create_detext_model(feature_type2name, task_type=TaskType.RANKING, **deep_match_param)
        outputs = detext_model.generate_training_scores(inputs)

        self.assertAllEqual(tf.shape(outputs), [2, 3])

    def testModelNetworkNoQueryNoDenseClassification(self):
        """Tests whether model can produce correct classification output when there's no dense features and no query feature"""
        num_classes = 7
        deep_match_param = copy.copy(self.deep_match_param)
        deep_match_param.num_classes = num_classes

        inputs = copy.copy(self.cls_inputs)
        inputs.pop(self.DENSE_FTRS_COLUMN_NAME_VAL)
        inputs.pop(self.QUERY_COLUMN_NAME_VAL)

        feature_type2name = copy.copy(self.feature_type2name)
        feature_type2name.pop(InputFtrType.QUERY_COLUMN_NAME)
        feature_type2name.pop(InputFtrType.DENSE_FTRS_COLUMN_NAMES)

        deep_match_param.use_dense_ftrs = False
        deep_match_param.has_query = False

        detext_model = model.create_detext_model(feature_type2name, task_type=TaskType.CLASSIFICATION, **deep_match_param)
        outputs = detext_model.generate_training_scores(inputs)

        self.assertAllEqual(outputs.shape, [2, num_classes])

    def testModelNetworkDenseOnlyClassification(self):
        """Tests whether model can produce correct classification output when there's only dense features"""
        num_classes = 7
        deep_match_param = copy.copy(self.deep_match_param)
        deep_match_param.num_classes = num_classes

        inputs = copy.copy(self.cls_inputs)
        inputs = {self.DENSE_FTRS_COLUMN_NAME_VAL: inputs[self.DENSE_FTRS_COLUMN_NAME_VAL]}

        feature_type2name = copy.copy(self.feature_type2name)
        feature_type2name = {InputFtrType.DENSE_FTRS_COLUMN_NAMES: feature_type2name[InputFtrType.DENSE_FTRS_COLUMN_NAMES]}

        deep_match_param.has_query = False
        deep_match_param.use_deep = False
        deep_match_param.use_sparse_ftrs = False

        detext_model = model.create_detext_model(feature_type2name, task_type=TaskType.CLASSIFICATION, **deep_match_param)
        outputs = detext_model.generate_training_scores(inputs)

        self.assertAllEqual(outputs.shape, [2, num_classes])

    def testModelNetworkSparseOnlyClassification(self):
        """Tests whether model can produce correct classification output when there's only sparse features"""
        num_classes = 7
        deep_match_param = copy.copy(self.deep_match_param)
        deep_match_param.num_classes = num_classes

        inputs = copy.copy(self.cls_inputs)
        inputs = {self.SPARSE_FTRS_COLUMN_NAME_VAL: inputs[self.SPARSE_FTRS_COLUMN_NAME_VAL]}

        feature_type2name = copy.copy(self.feature_type2name)
        feature_type2name = {InputFtrType.SPARSE_FTRS_COLUMN_NAMES: feature_type2name[InputFtrType.SPARSE_FTRS_COLUMN_NAMES]}

        deep_match_param.has_query = False
        deep_match_param.use_deep = False
        deep_match_param.use_dense_ftrs = False

        detext_model = model.create_detext_model(feature_type2name, task_type=TaskType.CLASSIFICATION, **deep_match_param)
        outputs = detext_model.generate_training_scores(inputs)

        self.assertAllEqual(outputs.shape, [2, num_classes])

    def testPadDenseFtrsNan(self):
        """Tests nan padding for dense features """
        nan_dense_ftrs = np.copy(self.ranking_dense_ftrs)
        nan_removed_dense_ftrs = np.copy(self.ranking_dense_ftrs)

        nan_dense_ftrs[0] = np.nan
        nan_removed_dense_ftrs[0] = 0
        dense_ftrs = tf.constant(nan_dense_ftrs, dtype=tf.float32)

        self.assertAllClose(model.DetextModel.pad_dense_ftrs_nan(dense_ftrs), nan_removed_dense_ftrs)

    def testCreateDetextModel(self):
        """Tests create_detext_model() """
        feature_type2name_lst = [self.feature_type2name]
        inputs_lst = [self.ranking_inputs]
        assert len(feature_type2name_lst) == len(inputs_lst)

        for feature_type2name, inputs in zip(feature_type2name_lst, inputs_lst):
            self._testCreateDetextModel(feature_type2name, inputs)

    def _testCreateDetextModel(self, feature_type2name, inputs):
        """Tests create_detext_model() given feature names and inputs"""
        model_dir = os.path.join(self.resource_dir, 'tmp_model')

        # Test model output shape
        detext_model = model.create_detext_model(feature_type2name, task_type=TaskType.RANKING, **self.deep_match_param)

        outputs = detext_model.generate_training_scores(inputs)
        self.assertAllEqual(tf.shape(outputs), [2, 3])

        # Test model export and loading
        detext_model.save(model_dir)
        loaded_model = tf.keras.models.load_model(model_dir)

        loaded_model_outputs = loaded_model.generate_training_scores(inputs)
        self.assertAllEqual(outputs, loaded_model_outputs)
        shutil.rmtree(model_dir)


if __name__ == '__main__':
    tf.test.main()
