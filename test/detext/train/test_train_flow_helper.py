import copy

import tensorflow as tf
from detext.train import data_fn, model, train_flow_helper
from detext.train.constant import Constant
from detext.utils.parsing_utils import HParams, InputFtrType, TaskType
from detext.utils.testing.data_setup import DataSetup


class TestTrainFlowHelper(tf.test.TestCase, DataSetup):
    """Unit test for train_flow_helper.py"""
    batch_size = 2
    nums_dense_ftrs = [3]
    task_type = TaskType.RANKING

    text_encoder_param = copy.copy(DataSetup.cnn_param)
    text_encoder_param.num_doc_fields = 1
    text_encoder_param.num_user_fields = 3

    id_encoder_param = copy.copy(DataSetup.id_encoder_param)
    id_encoder_param.num_id_fields = 2

    rep_layer_param = copy.copy(DataSetup.rep_layer_param)
    rep_layer_param.text_encoder_param = text_encoder_param
    rep_layer_param.id_encoder_param = id_encoder_param
    rep_layer_param.num_doc_fields = 1
    rep_layer_param.num_user_fields = 3

    rep_layer_param.num_doc_id_fields = 1
    rep_layer_param.num_user_id_fields = 1

    feature_type2name = {InputFtrType.QUERY_COLUMN_NAME: 'query',
                         InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_completedQuery'],
                         InputFtrType.DOC_ID_COLUMN_NAMES: ['docId_completedQuery'],
                         InputFtrType.USER_TEXT_COLUMN_NAMES: ['usr_headline', 'usr_skills', 'usr_currTitles'],
                         InputFtrType.USER_ID_COLUMN_NAMES: ['usrId_currTitles'],
                         InputFtrType.DENSE_FTRS_COLUMN_NAMES: ['wide_ftrs'],
                         InputFtrType.LABEL_COLUMN_NAME: 'label',
                         InputFtrType.WEIGHT_COLUMN_NAME: 'weight'}
    feature_name2num = {'wide_ftrs': nums_dense_ftrs[0]}

    deep_match_param = HParams(feature_name2num=feature_name2num,
                               use_dense_ftrs=True,
                               use_deep=True,
                               has_query=True,
                               use_sparse_ftrs=False,
                               sparse_embedding_cross_ftr_combiner='concat',
                               sparse_embedding_same_ftr_combiner='sum',
                               sparse_embedding_size=10,
                               emb_sim_func=['inner'],
                               rep_layer_param=rep_layer_param,
                               ftr_mean=None, ftr_std=None,
                               num_hidden=[3],
                               rescale_dense_ftrs=False,
                               num_classes=1,
                               task_ids=None)

    def testPredict(self):
        """Tests predict()"""
        dataset = data_fn.input_fn_tfrecord(input_pattern=self.data_dir,
                                            batch_size=self.batch_size,
                                            mode=tf.estimator.ModeKeys.EVAL,
                                            feature_type2name=self.feature_type2name,
                                            feature_name2num=self.feature_name2num,
                                            input_pipeline_context=None,
                                            )

        detext_model = model.create_detext_model(self.feature_type2name, task_type=self.task_type, **self.deep_match_param)
        predicted_output = train_flow_helper.predict_with_additional_info(dataset, detext_model, self.feature_type2name)

        for output in predicted_output:
            for key in [train_flow_helper._SCORES, self.feature_type2name.get(InputFtrType.WEIGHT_COLUMN_NAME, Constant()._DEFAULT_WEIGHT_FTR_NAME),
                        self.feature_type2name.get(InputFtrType.UID_COLUMN_NAME, Constant()._DEFAULT_UID_FTR_NAME),
                        self.feature_type2name[InputFtrType.LABEL_COLUMN_NAME]]:
                self.assertIn(key, output)


if __name__ == '__main__':
    tf.test.main()
