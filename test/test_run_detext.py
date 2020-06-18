"""Tests for run_detext."""
import os
import shutil
import tensorflow as tf
import sys

from detext.run_detext import main

root_dir = os.path.join(os.path.dirname(__file__), "resources")
out_dir = os.path.join(root_dir, "output")
data = os.path.join(root_dir, "train", "dataset", "tfrecord")
multitask_data = os.path.join(root_dir, "train", "multitask", "tfrecord")
vocab = os.path.join(root_dir, "vocab.txt")


class TestModel(tf.test.TestCase):
    base_args = ["--all_metrics", "precision@1", "ndcg@10",
                 "--emb_sim_func", "inner", "concat", "diff",
                 "--elem_rescale", "True",
                 "--explicit_empty", "False",
                 "--init_weight", "0.1",
                 "--lambda_metric", "None",
                 "--learning_rate", "0.002",
                 "--ltr_loss_fn", "softmax",
                 "--max_gradient_norm", "1.0",
                 "--max_len", "16",
                 "--min_len", "3",
                 "--num_filters", "50",
                 "--num_train_steps", "4",
                 "--num_units", "4",
                 "--optimizer", "bert_adam",
                 "--pmetric", "ndcg@10",
                 "--steps_per_stats", "1",
                 "--steps_per_eval", "2",
                 "--train_batch_size", "2",
                 "--test_batch_size", "2",
                 "--use_deep", "True",
                 "--vocab_file", vocab,
                 "--vocab_file_for_id_ftr", vocab,
                 "--resume_training", "False",
                 "--unknown_param", "unknown_value"]  # added unknown param to test arg parsing

    args = base_args + [
        "--feature_names", "label,query,doc_completedQuery,usr_headline,usr_skills,usr_currTitles,"
                           "usrId_currTitles,docId_completedQuery,wide_ftrs,weight",
        "--num_wide", "3",
        "--test_file", data,
        "--dev_file", data,
        "--train_file", data,
    ]

    multitask_args = base_args + [
        "--feature_names", "query,label,wide_ftrs,doc_field1,doc_field2,task_id",
        "--num_wide", "10",
        "--test_file", multitask_data,
        "--dev_file", multitask_data,
        "--train_file", multitask_data,
    ]

    def _cleanUp(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_run_detext_cnn(self):
        """
        This method test run_detext with CNN models
        """
        output = os.path.join(out_dir, "cnn_model")
        args = self.args + \
            ["--filter_window_size", "1", "2", "3",
             "--ftr_ext", "cnn",
             "--num_hidden", "10", "10", "5",
             "--out_dir", output]
        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_non_cnn(self):
        """
        This method test run_detext with non-CNN models
        """
        output = os.path.join(out_dir, "non_cnn_model")
        args = self.args + \
            ["--filter_window_size", "0",
             "--ftr_ext", "lstm",
             "--num_hidden", "10",
             "--out_dir", output]
        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)

    def test_run_detext_multitask(self):
        """
        This method test run_detext with multitasking models
        """
        output = os.path.join(out_dir, "multitask_model")
        args = self.multitask_args + \
            ["--filter_window_size", "3",
             "--ftr_ext", "cnn",
             "--num_hidden", "10",
             "--task_ids", "0,1",
             "--task_weights", "0.2,0.8",
             "--out_dir", output]
        sys.argv[1:] = args
        main(sys.argv)
        self._cleanUp(output)
