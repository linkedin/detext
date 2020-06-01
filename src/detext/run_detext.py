"""
Overall pipeline to train the model.  It parses arguments, and trains a DeText model.
"""

import argparse
import logging
import os
import sys
import time

import tensorflow as tf
import tensorflow_ranking as tfr
from detext.train import train
from detext.train.data_fn import input_fn
from detext.utils import misc_utils, logger, executor_utils


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--ftr_ext", choices=['cnn', 'bert', 'lstm_lm', 'lstm'], help="NLP feature extraction module.")
    parser.add_argument("--num_units", type=int, default=128, help="word embedding size.")
    parser.add_argument("--num_units_for_id_ftr", type=int, default=128, help="id feature embedding size.")
    parser.add_argument("--sp_emb_size", type=int, default=1, help="Embedding size of sparse features")
    parser.add_argument("--num_hidden", type=str, default='0', help="hidden size.")
    parser.add_argument("--num_wide", type=int, default=0, help="number of wide features per doc.")
    parser.add_argument("--num_wide_sp", type=int, default=None, help="number of sparse wide features per doc")
    parser.add_argument("--use_deep", type=str2bool, default=True, help="Whether to use deep features.")
    parser.add_argument("--elem_rescale", type=str2bool, default=True,
                        help="Whether to perform elementwise rescaling.")
    parser.add_argument("--use_doc_projection", type=str2bool, default=False,
                        help="whether to project multiple doc features to 1 vector.")
    parser.add_argument("--use_usr_projection", type=str2bool, default=False,
                        help="whether to project multiple usr features to 1 vector.")

    # Ranking specific
    parser.add_argument("--ltr_loss_fn", type=str, default='pairwise', help="learning-to-rank method.")
    parser.add_argument("--emb_sim_func", default='inner',
                        help="Approach to computing query/doc similarity scores: "
                             "inner/hadamard/concat or any combination of them separated by comma.")

    # Classification specific
    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of classes for multi-class classification tasks.")

    # CNN related
    parser.add_argument("--filter_window_sizes", type=str, default='3', help="CNN filter window sizes.")
    parser.add_argument("--num_filters", type=int, default=100, help="number of CNN filters.")
    parser.add_argument("--explicit_empty", type=str2bool, default=False,
                        help="Explicitly modeling empty string in cnn")

    # BERT related
    parser.add_argument("--lr_bert", type=float, default=None, help="Learning rate factor for bert components")
    parser.add_argument("--bert_config_file", type=str, default=None, help="bert config.")
    parser.add_argument("--bert_checkpoint", type=str, default=None, help="pretrained bert model checkpoint.")

    # LSTM related
    parser.add_argument("--unit_type", type=str, default="lstm", choices=["lstm"],
                        help="RNN cell unit type. Currently only supports lstm. Will support other cell types in the future")
    parser.add_argument("--num_layers", type=int, default=1, help="RNN layers")
    parser.add_argument("--num_residual_layers", type=int, default=0,
                        help="Number of residual layers from top to bottom. For example, if `num_layers=4` and "
                             "`num_residual_layers=2`, the last 2 RNN cells in the returned list will be wrapped "
                             "with `ResidualWrapper`.")
    parser.add_argument("--forget_bias", type=float, default=1., help="Forget bias of RNN cell")
    parser.add_argument("--rnn_dropout", type=float, default=0., help="Dropout of RNN cell")
    parser.add_argument("--bidirectional", type=str2bool, default=False, help="Whether to use bidirectional RNN")
    parser.add_argument("--normalized_lm", type=str2bool, default=False,
                        help="Whether to use normalized lm. This option only works for unit_type=lstm_lm")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "bert_adam", "bert_lamb"], default="sgd",
                        help="Type of optimizer to use. bert_adam is similar to the optimizer implementation in bert.")
    parser.add_argument("--max_gradient_norm", type=float, default=1.0, help="Clip gradients to this norm.")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--num_train_steps", type=int, default=1, help="Num steps to train.")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Num of epochs to train, will overwrite train_steps if set")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Num steps for warmup.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training data batch size.")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Test data batch size.")
    parser.add_argument("--l1", type=float, default=None, help="Scale of L1 regularization")
    parser.add_argument("--l2", type=float, default=None, help="Scale of L2 regularization")

    # Data
    parser.add_argument("--train_file", type=str, default=None, help="Train file.")
    parser.add_argument("--dev_file", type=str, default=None, help="Dev file.")
    parser.add_argument("--test_file", type=str, default=None, help="Test file.")
    parser.add_argument("--out_dir", type=str, default=None, help="Store log/model files.")
    parser.add_argument("--std_file", type=str, default=None, help="feature standardization file")
    parser.add_argument("--max_len", type=int, default=32, help="max sent length.")
    parser.add_argument("--min_len", type=int, default=3, help="min sent length.")

    # Vocab and word embedding
    parser.add_argument("--vocab_file", type=str, default=None, help="Vocab file")
    parser.add_argument("--we_file", type=str, default=None, help="Pretrained word embedding file")
    parser.add_argument("--we_trainable", type=str2bool, default=True, help="Whether to train word embedding")
    parser.add_argument("--PAD", type=str, default="[PAD]", help="Token for padding")
    parser.add_argument("--SEP", type=str, default="[SEP]", help="Token for sentence separation")
    parser.add_argument("--CLS", type=str, default="[CLS]", help="Token for start of sentence")
    parser.add_argument("--UNK", type=str, default="[UNK]", help="Token for unknown word")
    parser.add_argument("--MASK", type=str, default="[MASK]", help="Token for masked word")

    # Vocab and word embedding for id features
    parser.add_argument("--vocab_file_for_id_ftr", type=str, default=None, help="Vocab file for id features")
    parser.add_argument("--we_file_for_id_ftr", type=str, default=None,
                        help="Pretrained word embedding file for id features")
    parser.add_argument("--we_trainable_for_id_ftr", type=str2bool, default=True,
                        help="Whether to train word embedding for id features")
    parser.add_argument("--PAD_FOR_ID_FTR", type=str, default="[PAD]", help="Padding token for id features")
    parser.add_argument("--UNK_FOR_ID_FTR", type=str, default="[UNK]", help="Unknown word token for id features")

    # Misc
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed (>0, set a specific seed).")
    parser.add_argument("--steps_per_stats", type=int, default=100, help="training steps to print statistics.")
    parser.add_argument("--num_eval_rounds", type=int, default=None, help="number of evaluation round,this param will "
                                                                          "override steps_per_eval as max(1,"
                                                                          "num_train_steps / num_eval_rounds)")
    parser.add_argument("--steps_per_eval", type=int, default=1000, help="training steps to evaluate datasets.")
    parser.add_argument("--keep_checkpoint_max", type=int, default=5,
                        help="The maximum number of recent checkpoint files to keep. If 0, all checkpoint "
                             "files are kept. Defaults to 5")
    parser.add_argument("--feature_names", type=str, default=None, help="the feature names.")
    parser.add_argument("--lambda_metric", type=str, default=None, help="only support ndcg.")
    parser.add_argument("--init_weight", type=float, default=0.1, help="weight initialization value.")
    parser.add_argument("--pmetric", type=str, default=None, help="Primary metric.")
    parser.add_argument("--all_metrics", type=str, default=None, help="All metrics.")
    parser.add_argument("--score_rescale", type=str, default=None, help="The mean and std of previous model.")
    parser.add_argument("--tokenization", type=str, default='punct', choices=['plain', 'punct'],
                        help="The tokenzation performed for data preprocessing. "
                             "Currently support: punct/plain(no split). "
                             "Note that this should be set correctly to ensure consistency for savedmodel.")

    parser.add_argument("--resume_training", type=str2bool, default=False,
                        help="Whether to resume training from checkpoint in out_dir.")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="The metadata_path for converted avro2tf avro data.")

    # tf-ranking related
    parser.add_argument("--use_tfr_loss", type=str2bool, default=False, help="whether to use tf-ranking loss.")
    parser.add_argument('--tfr_loss_fn',
                        choices=[
                            tfr.losses.RankingLossKey.SOFTMAX_LOSS,
                            tfr.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS],
                        default=tfr.losses.RankingLossKey.SOFTMAX_LOSS,
                        help="softmax_loss")
    parser.add_argument('--tfr_lambda_weights', type=str, default=None)

    parser.add_argument('--use_horovod', type=str2bool, default=False,
                        help="whether to use horovod for sync distributed training")

    # multitask training related
    parser.add_argument('--task_ids', type=str, default=None,
                        help="All types of task IDs for multitask training. E.g., 1,2,3")
    parser.add_argument('--task_weights', type=str, default=None,
                        help="Weights for each task specified in task_ids. E.g., 0.5,0.3,0.2")


def str2bool(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_hparams(flags):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        # Data
        ftr_ext=flags.ftr_ext,
        filter_window_sizes=flags.filter_window_sizes,
        num_units=flags.num_units,
        sp_emb_size=flags.sp_emb_size,
        num_units_for_id_ftr=flags.num_units_for_id_ftr,
        num_filters=flags.num_filters,
        num_hidden=flags.num_hidden,
        num_wide=flags.num_wide,
        ltr_loss_fn=flags.ltr_loss_fn,
        use_deep=flags.use_deep,
        elem_rescale=flags.elem_rescale,
        emb_sim_func=flags.emb_sim_func,
        use_doc_projection=flags.use_doc_projection,
        use_usr_projection=flags.use_usr_projection,
        num_classes=flags.num_classes,
        optimizer=flags.optimizer,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        lr_bert=flags.lr_bert,
        num_train_steps=flags.num_train_steps,
        num_epochs=flags.num_epochs,
        num_warmup_steps=flags.num_warmup_steps,
        train_batch_size=flags.train_batch_size,
        test_batch_size=flags.test_batch_size,
        train_file=flags.train_file,
        dev_file=flags.dev_file,
        test_file=flags.test_file,
        out_dir=flags.out_dir,
        vocab_file=flags.vocab_file,
        we_file=flags.we_file,
        we_trainable=flags.we_trainable,
        random_seed=flags.random_seed,
        steps_per_stats=flags.steps_per_stats,
        steps_per_eval=flags.steps_per_eval,
        num_eval_rounds=flags.num_eval_rounds,
        keep_checkpoint_max=flags.keep_checkpoint_max,
        max_len=flags.max_len,
        min_len=flags.min_len,
        feature_names=flags.feature_names,
        lambda_metric=flags.lambda_metric,
        bert_config_file=flags.bert_config_file,
        bert_checkpoint=flags.bert_checkpoint,
        init_weight=flags.init_weight,
        pmetric=flags.pmetric,
        std_file=flags.std_file,
        num_wide_sp=flags.num_wide_sp,
        all_metrics=flags.all_metrics,
        score_rescale=flags.score_rescale,
        explicit_empty=flags.explicit_empty,
        tokenization=flags.tokenization,
        unit_type=flags.unit_type,
        num_layers=flags.num_layers,
        num_residual_layers=flags.num_residual_layers,
        forget_bias=flags.forget_bias,
        rnn_dropout=flags.rnn_dropout,
        bidirectional=flags.bidirectional,
        PAD=flags.PAD,
        SEP=flags.SEP,
        CLS=flags.CLS,
        UNK=flags.UNK,
        MASK=flags.MASK,
        resume_training=flags.resume_training,
        metadata_path=flags.metadata_path,
        tfr_loss_fn=flags.tfr_loss_fn,
        tfr_lambda_weights=flags.tfr_lambda_weights,
        use_tfr_loss=flags.use_tfr_loss,
        use_horovod=flags.use_horovod,
        normalized_lm=flags.normalized_lm,
        task_ids=flags.task_ids,
        task_weights=flags.task_weights,

        # Vocab and word embedding for id features
        PAD_FOR_ID_FTR=flags.PAD_FOR_ID_FTR,
        UNK_FOR_ID_FTR=flags.UNK_FOR_ID_FTR,
        vocab_file_for_id_ftr=flags.vocab_file_for_id_ftr,
        we_file_for_id_ftr=flags.we_file_for_id_ftr,
        we_trainable_for_id_ftr=flags.we_trainable_for_id_ftr,

        l1=flags.l1,
        l2=flags.l2,
    )


def get_hparams(argv):
    """
    Get hyper-parameters.
    """
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    hparams, unknown_params = parser.parse_known_args(argv)
    hparams = create_hparams(hparams)

    # Print all hyper-parameters
    for k, v in sorted(vars(hparams).items()):
        print('--' + k + '=' + str(v))

    return hparams


def main(argv):
    """
    This is the main method for training the model.
    :param argv: training parameters
    :return:
    """
    # Get executor task type from TF_CONFIG
    task_type = executor_utils.get_executor_task_type()

    # Get hyper-parameters.
    hparams = get_hparams(argv)

    tf.logging.set_verbosity(tf.logging.INFO)

    # if epoch is set, overwrite training steps
    if hparams.num_epochs is not None:
        hparams.num_train_steps = misc_utils.estimate_train_steps(
            hparams.train_file,
            hparams.num_epochs,
            hparams.train_batch_size,
            hparams.metadata_path is None)

    # if num_eval_rounds is set, override steps_per_eval
    if hparams.num_eval_rounds is not None:
        hparams.steps_per_eval = max(1, int(hparams.num_train_steps / hparams.num_eval_rounds))

    # Create directory and launch tensorboard
    if task_type == executor_utils.CHIEF or task_type == executor_utils.LOCAL_MODE:
        # If not resume training from checkpoints, delete output directory.
        if not hparams.resume_training:
            logging.info("Removing previous output directory...")
            if tf.gfile.Exists(hparams.out_dir):
                tf.gfile.DeleteRecursively(hparams.out_dir)

        # If output directory deleted or does not exist, create the directory.
        if not tf.gfile.Exists(hparams.out_dir):
            logging.info('Creating dirs recursively at: {0}'.format(hparams.out_dir))
            tf.gfile.MakeDirs(hparams.out_dir)

        misc_utils.save_hparams(hparams.out_dir, hparams)

    else:
        # TODO: move removal/creation to a hadoopShellJob st. it does not reside in distributed training code.
        logging.info("Waiting for chief to remove/create directories.")
        # Wait for dir created form chief
        time.sleep(10)

    if task_type == executor_utils.EVALUATOR or task_type == executor_utils.LOCAL_MODE:
        # set up logger for evaluator
        sys.stdout = logger.Logger(os.path.join(hparams.out_dir, 'eval_log.txt'))

    hparams = misc_utils.extend_hparams(hparams)

    logging.info("***********DeText Training***********")

    # Train and evaluate DeText model
    train.train(hparams, input_fn)


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main)
