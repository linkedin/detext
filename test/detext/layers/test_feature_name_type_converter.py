from collections import OrderedDict

import tensorflow as tf

from detext.layers import feature_name_type_converter
from detext.utils.parsing_utils import InputFtrType, TaskType, InternalFtrType
from detext.utils.testing.data_setup import DataSetup


class TestFeatureTypeNameConverter(tf.test.TestCase, DataSetup):
    """Unit test for feature_name_type_converter.py"""
    ranking_dense_ftrs = tf.random.uniform(shape=[2, 3, 5])
    cls_dense_ftrs = tf.random.uniform(shape=[2, 5])

    ranking_inputs = OrderedDict(sorted({'query': DataSetup.query,
                                         'usr_headline': DataSetup.user_fields[0],
                                         'usr_title': DataSetup.user_fields[1],
                                         'usrId_headline': DataSetup.user_id_fields[0],
                                         'usrId_title': DataSetup.user_id_fields[1],
                                         'doc_headline': DataSetup.ranking_doc_fields[0],
                                         'doc_title': DataSetup.ranking_doc_fields[1],
                                         'docId_headline': DataSetup.ranking_doc_id_fields[0],
                                         'docId_title': DataSetup.ranking_doc_id_fields[1],
                                         'dense_ftrs': ranking_dense_ftrs,
                                         'task_id_field': tf.constant([2, 1])
                                         }.items()))

    classification_inputs = OrderedDict(sorted({'query': DataSetup.query,
                                                'usr_headline': DataSetup.user_fields[0],
                                                'usr_title': DataSetup.user_fields[1],
                                                'usrId_headline': DataSetup.user_id_fields[0],
                                                'usrId_title': DataSetup.user_id_fields[1],

                                                'doc_headline': DataSetup.cls_doc_fields[0],
                                                'doc_title': DataSetup.cls_doc_fields[1],
                                                'docId_headline': DataSetup.cls_doc_id_fields[0],
                                                'docId_title': DataSetup.cls_doc_id_fields[1],
                                                'dense_ftrs': cls_dense_ftrs,
                                                'task_id_field': tf.constant([2, 1])
                                                }.items()))

    feature_type2name_deep = {InputFtrType.QUERY_COLUMN_NAME: 'query',
                              InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_headline', 'doc_title'],
                              InputFtrType.DOC_ID_COLUMN_NAMES: ['docId_headline', 'docId_title'],
                              InputFtrType.USER_TEXT_COLUMN_NAMES: ['usr_headline', 'usr_title'],
                              InputFtrType.USER_ID_COLUMN_NAMES: ['usrId_headline', 'usrId_title']}
    feature_type2name_wide = {InputFtrType.DENSE_FTRS_COLUMN_NAMES: ['dense_ftrs']}
    feature_type2name_multitask = {InputFtrType.TASK_ID_COLUMN_NAME: 'task_id_field'}

    feature_type2name = {**feature_type2name_deep, **feature_type2name_wide, **feature_type2name_multitask}

    def test_name_type_converter_ranking(self):
        task_type_list = [TaskType.RANKING, TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION]
        inputs_list = [self.ranking_inputs, self.classification_inputs, self.classification_inputs]
        for task_type, inputs in zip(task_type_list, inputs_list):
            self._test_name_type_converter(task_type, inputs)

    def _test_name_type_converter(self, task_type, inputs):
        converter = feature_name_type_converter.FeatureNameTypeConverter(task_type=task_type, feature_type2name=self.feature_type2name)
        converter_type_list = [InternalFtrType.DEEP_FTR_BAG, InternalFtrType.WIDE_FTR_BAG, InternalFtrType.MULTITASK_FTR_BAG]
        result_feature_type2name_list = [self.feature_type2name_deep, self.feature_type2name_wide, self.feature_type2name_multitask]
        for converter_type, result_feature_type2_name in zip(converter_type_list, result_feature_type2name_list):
            outputs = converter({converter_type: inputs})[converter_type]
            self.assertAllEqual(sorted(outputs.keys()), sorted(result_feature_type2_name.keys()))

            # Classification expands an additional dimension so that cls inputs can reuse the ranking layers since it's treated as list_size=1
            if task_type in [TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION]:
                if converter_type == InternalFtrType.WIDE_FTR_BAG:
                    for t in outputs[InputFtrType.DENSE_FTRS_COLUMN_NAMES]:
                        self.assertAllEqual(tf.rank(t), 3)
                if converter_type == InternalFtrType.DEEP_FTR_BAG:
                    for t in outputs[InputFtrType.DOC_TEXT_COLUMN_NAMES]:
                        self.assertAllEqual(tf.rank(t), 2)
                    for t in outputs[InputFtrType.DOC_ID_COLUMN_NAMES]:
                        self.assertAllEqual(tf.rank(t), 2)


if __name__ == '__main__':
    tf.test.main()
