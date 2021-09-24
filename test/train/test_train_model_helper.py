import os
import shutil
from collections import OrderedDict

import tensorflow as tf

from detext.train import model
from detext.train import train_model_helper
from detext.utils import vocab_utils
from detext.utils.parsing_utils import HParams, InputFtrType, TaskType
from detext.utils.testing.data_setup import DataSetup


class TestTrainModelHelper(tf.test.TestCase, DataSetup):
    """Tests train_model_helper.py """
    task_id = tf.constant([1, 0])
    nums_dense_ftrs = [5]
    nums_sparse_ftrs = []
    sparse_ftrs_idx = sparse_ftrs_val = task_id_field = None
    task_type = TaskType.RANKING

    # Dense features and labels
    dense_ftrs = tf.random.uniform(shape=[2, 3, nums_dense_ftrs[0]])
    labels = tf.constant([[1, 0, 1], [0, 0, 1]])

    # Config and hparams
    embedding_layer_param = DataSetup.embedding_layer_param
    vocab_layer_param = DataSetup.vocab_layer_param
    text_encoder_param = DataSetup.cnn_param
    id_encoder_param = DataSetup.id_encoder_param
    rep_layer_param = DataSetup.rep_layer_param

    inputs = OrderedDict(sorted({'query': DataSetup.query,
                                 'usr_headline': DataSetup.user_fields[0],
                                 'usr_title': DataSetup.user_fields[1],
                                 'usrId_headline': DataSetup.user_id_fields[0],
                                 'usrId_title': DataSetup.user_id_fields[1],
                                 'doc_headline': DataSetup.ranking_doc_fields[0],
                                 'doc_title': DataSetup.ranking_doc_fields[1],
                                 'docId_headline': DataSetup.ranking_doc_id_fields[0],
                                 'docId_title': DataSetup.ranking_doc_id_fields[1],
                                 'dense_ftrs': dense_ftrs
                                 }.items()))

    feature_type2name = {InputFtrType.QUERY_COLUMN_NAME: 'query',
                         InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_headline', 'doc_title'],
                         InputFtrType.DOC_ID_COLUMN_NAMES: ['docId_headline', 'docId_title'],
                         InputFtrType.USER_TEXT_COLUMN_NAMES: ['usr_headline', 'usr_title'],
                         InputFtrType.USER_ID_COLUMN_NAMES: ['usrId_headline', 'usrId_title'],
                         InputFtrType.DENSE_FTRS_COLUMN_NAMES: ['dense_ftrs']}
    feature_name2num = {'dense_ftrs': nums_dense_ftrs[0]}

    deep_match_param = HParams(use_dense_ftrs=True,
                               use_deep=True,
                               has_query=True,
                               use_sparse_ftrs=False,
                               sparse_embedding_size=10,
                               sparse_embedding_cross_ftr_combiner='concat',
                               sparse_embedding_same_ftr_combiner='sum',
                               emb_sim_func=['inner'],
                               rep_layer_param=rep_layer_param,
                               ftr_mean=None, ftr_std=None,
                               num_hidden=[3],
                               rescale_dense_ftrs=False,
                               num_classes=1,
                               task_ids=None)

    hparams = HParams(ltr_loss_fn='softmax', l1=0.1, l2=0.2, use_tfr_loss=False, tfr_loss_fn=None, tfr_lambda_weights=None, lr_bert=0.01, learning_rate=0.1,
                      feature_type2name=feature_type2name,
                      feature_name2num=feature_name2num,
                      num_units_for_id_ftr=DataSetup.num_units_for_id_ftr,
                      vocab_hub_url_for_id_ftr='',
                      we_file_for_id_ftr='',
                      we_trainable_for_id_ftr=True,
                      task_type=task_type,
                      vocab_file=DataSetup.vocab_file,
                      vocab_file_for_id_ftr=DataSetup.vocab_file_for_id_ftr,
                      CLS=DataSetup.CLS, PAD=DataSetup.PAD, SEP=DataSetup.SEP, UNK=DataSetup.UNK,
                      PAD_FOR_ID_FTR=DataSetup.PAD_FOR_ID_FTR, UNK_FOR_ID_FTR=DataSetup.UNK_FOR_ID_FTR,
                      max_filter_window_size=max(DataSetup.filter_window_sizes),
                      **{**deep_match_param, **text_encoder_param, **rep_layer_param, **id_encoder_param, **embedding_layer_param})

    def testGetFuncParamFromHparams(self):
        """Tests get_func_param_from_hparams() """

        def test_function(param1, param2, exclude_param):
            return param1 - param2 + exclude_param

        hparams = train_model_helper.HParams(param1=1, param2=2, other_param=4)
        extracted_params = train_model_helper._get_func_param_from_hparams(test_function, hparams, exclude_lst=('exclude_param',))
        expected_params = train_model_helper.HParams(param1=1, param2=2)
        self.assertEqual(extracted_params, expected_params)

    def test(self):
        """Tests runnability of functions in train_model_helper.py """
        train_model_helper.get_optimizer_fn(self.hparams)
        train_model_helper.get_bert_optimizer_fn(self.hparams)
        train_model_helper.get_model_fn(self.hparams)
        train_model_helper.get_loss_fn(self.hparams)

        train_model_helper._get_rep_layer_param(self.hparams)
        train_model_helper._get_text_encoder_param(self.hparams)

    def testGetModelInput(self):
        """Tests get_model_input() """
        inputs = {'doc_header': tf.constant(1.0), 'doc_note': tf.constant(1.0), 'label': tf.constant(2.0)}
        feature_type2name = {InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_header'], InputFtrType.LABEL_COLUMN_NAME: 'label'}
        model_inputs = train_model_helper.get_model_input(inputs, feature_type2name=feature_type2name)
        self.assertAllEqual(model_inputs, {'doc_header': tf.constant(1.0)})

    def testGetWeight(self):
        """Tests get_weight() """
        feature_type2name = self.feature_type2name.copy()
        feature_type2name[InputFtrType.WEIGHT_COLUMN_NAME] = InputFtrType.WEIGHT_COLUMN_NAME
        feature_type2name[InputFtrType.TASK_ID_COLUMN_NAME] = InputFtrType.TASK_ID_COLUMN_NAME
        weight = train_model_helper.get_weight({feature_type2name[InputFtrType.TASK_ID_COLUMN_NAME]: tf.constant([0, 1])},
                                               {feature_type2name[InputFtrType.WEIGHT_COLUMN_NAME]: tf.constant([1.0, 2.0])},
                                               feature_type2name=feature_type2name,
                                               task_ids=[0, 1, 2],
                                               task_weights=[1.0, 2.0, 3.0])
        self.assertAllEqual(weight, tf.constant([1.0, 4.0]))

    def testLoadModelWithCkpt(self):
        """Tests load model with checkpoints"""
        ckpt_dir = os.path.join(self.resource_dir, 'tmp_ckpt')
        ckpt_path = os.path.join(ckpt_dir, 'tmp_model')

        detext_model = model.create_detext_model(self.feature_type2name, self.feature_name2num, task_type=self.task_type, **self.deep_match_param)
        outputs = detext_model.generate_training_scores(self.inputs)
        original_outputs = outputs

        # Test model export and loading
        detext_model.save_weights(ckpt_path)
        loaded_model = train_model_helper.load_model_with_ckpt(self.hparams, ckpt_path)

        loaded_model_outputs = loaded_model.generate_training_scores(self.inputs)
        self.assertAllEqual(original_outputs, loaded_model_outputs)
        shutil.rmtree(ckpt_dir)

    def testGetInputFnCommon(self):
        """Tests get_input_fn_common"""
        feature_type2name = {InputFtrType.QUERY_COLUMN_NAME: 'query',
                             InputFtrType.DOC_TEXT_COLUMN_NAMES: ['doc_completedQuery'],
                             InputFtrType.DOC_ID_COLUMN_NAMES: ['docId_completedQuery'],
                             InputFtrType.USER_TEXT_COLUMN_NAMES: ['usr_headline', 'usr_skills', 'usr_currTitles'],
                             InputFtrType.USER_ID_COLUMN_NAMES: ['usrId_currTitles'],
                             InputFtrType.DENSE_FTRS_COLUMN_NAMES: 'wide_ftrs',
                             InputFtrType.LABEL_COLUMN_NAME: 'label',
                             InputFtrType.WEIGHT_COLUMN_NAME: 'weight'}
        feature_name2num = {'wide_ftrs': 5}

        _, vocab_tf_table = vocab_utils.read_tf_vocab(self.vocab_file, self.UNK)
        vocab_table = vocab_utils.read_vocab(self.vocab_file)
        data_dir = self.data_dir
        hparams = HParams(input_pattern=data_dir,
                          filter_window_sizes=[10],
                          CLS=self.CLS, PAD=self.PAD, SEP=self.SEP, UNK=self.UNK, UNK_FOR_ID_FTR=self.UNK, PAD_FOR_ID_FTR=self.PAD,
                          min_len=1, max_len=16,
                          vocab_file=self.vocab_file, vocab_file_for_id_ftr=self.vocab_file,
                          PAD_ID=vocab_table[self.PAD], SEP_ID=vocab_table[self.SEP], CLS_ID=vocab_table[self.CLS],
                          mode=tf.estimator.ModeKeys.EVAL,
                          task_type=self.task_type,
                          vocab_table=vocab_tf_table, vocab_table_for_id_ftr=vocab_tf_table,
                          max_filter_window_size=3,
                          vocab_hub_url='',
                          vocab_hub_url_for_id_ftr='',
                          embedding_hub_url='',
                          embedding_hub_url_for_id_ftr='',
                          feature_type2name=feature_type2name,
                          feature_name2num=feature_name2num)
        train_model_helper.get_input_fn_common(data_dir, 1, tf.estimator.ModeKeys.TRAIN, hparams)


if __name__ == '__main__':
    tf.test.main()
