"""Tests for run_detext."""
import os
import shutil
import sys
from pathlib import Path
from subprocess import run, PIPE

import tensorflow as tf
from detext.run_detext import main
from detext.utils.parsing_utils import InputFtrType, TaskType
from detext.utils.testing.data_setup import DataSetup


class TestRunDeText(tf.test.TestCase, DataSetup):
    base_args = ["--all_metrics", "precision@1", "ndcg@10",
                 "--emb_sim_func", "inner", "concat", "diff",
                 "--rescale_dense_ftrs", "True",
                 "--learning_rate", "0.002",
                 "--ltr_loss_fn", "softmax",
                 "--max_len", "16",
                 "--min_len", "3",
                 "--num_filters", "5",
                 "--num_train_steps", "4",
                 "--num_units", "10",
                 "--optimizer", "adamw",
                 "--pmetric", "ndcg@10",
                 "--steps_per_stats", "1",
                 "--steps_per_eval", "2",
                 "--train_batch_size", "2",
                 "--test_batch_size", "2",
                 "--vocab_file_for_id_ftr", DataSetup.vocab_file,
                 "--distribution_strategy", "one_device",
                 "--num_gpu", "0",
                 "--run_eagerly", "False",
                 "--ftr_ext", "cnn",
                 "--filter_window_size", "3",
                 "--unknown_param", "unknown_value"
                 ]  # added unknown param to test arg parsing

    ranking_feature_args = [
        f"--{InputFtrType.LABEL_COLUMN_NAME}", "label",
        f"--{InputFtrType.QUERY_COLUMN_NAME}", "query",
        f"--{InputFtrType.DOC_TEXT_COLUMN_NAMES}", "doc_title",
        f"--{InputFtrType.USER_TEXT_COLUMN_NAMES}", "user_title", "user_headline",
        f"--{InputFtrType.DOC_ID_COLUMN_NAMES}", "doc_headline_id",
        f"--{InputFtrType.DENSE_FTRS_COLUMN_NAMES}", "dense_ftrs", "dense_ftrs_2",
        "--nums_dense_ftrs", "2", "2",
        f"--{InputFtrType.SPARSE_FTRS_COLUMN_NAMES}", "sparse_ftrs",
        f"--{InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES}", "sparse_ftrs", "sparse_ftrs1",
        "--sparse_embedding_cross_ftr_combiner", "concat",
        "--nums_sparse_ftrs", "10",
        "--nums_shallow_tower_sparse_ftrs", "10", "10",
        f"--{InputFtrType.WEIGHT_COLUMN_NAME}", "weight",
    ]

    ranking_args = base_args + ranking_feature_args + [
        "--vocab_hub_url", DataSetup.vocab_hub_url,
        "--test_file", DataSetup.ranking_data_dir,
        "--dev_file", DataSetup.ranking_data_dir,
        "--train_file", DataSetup.ranking_data_dir,
    ]

    multitask_feature_args = [
        f"--{InputFtrType.QUERY_COLUMN_NAME}", "query",
        f"--{InputFtrType.LABEL_COLUMN_NAME}", "label",
        f"--{InputFtrType.DENSE_FTRS_COLUMN_NAMES}", "wide_ftrs",
        f"--{InputFtrType.DOC_TEXT_COLUMN_NAMES}", "doc_field1", "doc_field2",
        f"--{InputFtrType.TASK_ID_COLUMN_NAME}", "task_id",
    ]

    multitask_args = base_args + multitask_feature_args + [
        "--nums_dense_ftrs", "10",
        "--vocab_hub_url", DataSetup.vocab_hub_url,
        "--test_file", DataSetup.multitask_data_dir,
        "--dev_file", DataSetup.multitask_data_dir,
        "--train_file", DataSetup.multitask_data_dir,
    ]

    bert_args = base_args + ranking_feature_args + [
        "--test_file", DataSetup.ranking_data_dir,
        "--dev_file", DataSetup.ranking_data_dir,
        "--train_file", DataSetup.ranking_data_dir,
        "--ftr_ext", "bert",
        "--num_units", "10"
    ]

    def _cleanUp(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_run_detext_bert_ranking(self):
        """
        This method test run_detext with BERT models
        """
        output = os.path.join(DataSetup.out_dir, "bert_model")
        self._cleanUp(output)
        args = self.bert_args + [
            "--bert_hub_url", DataSetup.bert_hub_url,  # pretrained bert hidden size: 16
            "--out_dir", output]

        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_libert_sp_ranking(self):
        """
        This method test run_detext with LiBERT sentencepiece tokenization models
        """
        output = os.path.join(DataSetup.out_dir, "libert_sp_model")
        self._cleanUp(output)
        args = self.bert_args + [
            "--bert_hub_url", DataSetup.libert_sp_hub_url,  # pretrained bert hidden size: 16
            "--out_dir", output]

        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_libert_space_ranking(self):
        """
        This method test run_detext with LiBERT space tokenization models
        """
        output = os.path.join(DataSetup.out_dir, "libert_space_model")
        self._cleanUp(output)
        args = self.bert_args + [
            "--bert_hub_url", DataSetup.libert_space_hub_url,  # pretrained bert hidden size: 16
            "--out_dir", output]

        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_cnn_ranking(self):
        """
        This method test run_detext with CNN models
        """
        output = os.path.join(DataSetup.out_dir, "cnn_model")
        self._cleanUp(output)
        args = self.ranking_args + ["--out_dir", output]
        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_multitask_ranking(self):
        """
        This method test run_detext with multitasking models
        """
        output = os.path.join(DataSetup.out_dir, "multitask_model")
        args = self.multitask_args + ["--task_ids", "0", "1",
                                      "--task_weights", "0.2", "0.8",
                                      "--out_dir", output]
        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_libert_binary_classification(self):
        """
        This method tests run_detext for libert binary classification fine-tuning
        """
        output = os.path.join(DataSetup.out_dir, "cls_libert_model")
        args = self.base_args + [
            "--task_type", TaskType.BINARY_CLASSIFICATION,
            "--ftr_ext", "bert",
            "--lr_bert", "0.00001",
            "--bert_hub_url", DataSetup.libert_sp_hub_url,
            "--num_units", "16",
            f"--{InputFtrType.LABEL_COLUMN_NAME}", "label",
            f"--{InputFtrType.DENSE_FTRS_COLUMN_NAMES}", "dense_ftrs",
            "--nums_dense_ftrs", "8",
            f"--{InputFtrType.SPARSE_FTRS_COLUMN_NAMES}", "sparse_ftrs",
            "--nums_sparse_ftrs", "30",
            f"--{InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES}", "sparse_ftrs",
            "--nums_shallow_tower_sparse_ftrs", "30",
            "--pmetric", "auc",
            "--all_metrics", "accuracy", "auc",
            "--test_file", DataSetup.binary_cls_data_dir,
            "--dev_file", DataSetup.binary_cls_data_dir,
            "--train_file", DataSetup.binary_cls_data_dir,
            "--out_dir", output]
        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_libert_classification(self):
        """
        This method tests run_detext for libert classification fine-tuning
        """
        output = os.path.join(DataSetup.out_dir, "cls_libert_model")
        args = self.base_args + [
            "--task_type", "classification",
            "--ftr_ext", "bert",
            "--lr_bert", "0.00001",
            "--bert_hub_url", DataSetup.libert_sp_hub_url,
            "--num_units", "16",
            f"--{InputFtrType.LABEL_COLUMN_NAME}", "label",
            f"--{InputFtrType.DOC_TEXT_COLUMN_NAMES}", "query_text",
            f"--{InputFtrType.USER_TEXT_COLUMN_NAMES}", "user_headline",
            f"--{InputFtrType.DENSE_FTRS_COLUMN_NAMES}", "dense_ftrs",
            f"--{InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES}", "sparse_ftrs",
            "--nums_shallow_tower_sparse_ftrs", "30",
            "--nums_dense_ftrs", "8",
            "--num_classes", "6",
            "--pmetric", "accuracy",
            "--all_metrics", "accuracy", "confusion_matrix",
            "--test_file", DataSetup.cls_data_dir,
            "--dev_file", DataSetup.cls_data_dir,
            "--train_file", DataSetup.cls_data_dir,
            "--out_dir", output]
        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def _test_demo(self):
        # TODO: enable this test when move this script to open source DeText
        completed_process = run(['sh', 'run_detext.sh'], stderr=PIPE, cwd=f'{Path(__file__).parent}/resources')
        self._cleanUp('/tmp/detext-output/hc_cnn_f50_u32_h100')
        self.assertEqual(completed_process.returncode, 0)


if __name__ == '__main__':
    tf.test.main()
