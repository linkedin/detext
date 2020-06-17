"""
Overall pipeline to train the model.  It parses arguments, and trains a DeText model.
"""

import logging
import os
import sys
import time
from typing import NamedTuple, List

import tensorflow as tf
import tensorflow_ranking as tfr
from arg_suite.arg_suite import arg_suite
from detext.train import train
from detext.train.data_fn import input_fn
from detext.utils import misc_utils, logger, executor_utils


@arg_suite
class Args(NamedTuple):
    """
    DeText: a Deep Text understanding framework for NLP related ranking, classification, and language generation tasks.
    It leverages semantic matching using deep neural networks to
    understand member intents in search and recommender systems.
    As a general NLP framework, currently DeText can be applied to many tasks,
    including search & recommendation ranking, multi-class classification and query understanding tasks.
    """
    #  kwargs utility funcation for split a str to a List, provided for backward compatibilities
    #  Please use built-in List parsing support when adding new arguments
    __comma_split_list = lambda t: {'type': lambda s: [t(item) for item in s.split(',')] if ',' in s else t(s), 'nargs': None}  # noqa: E731
    # network
    ftr_ext: str  # NLP feature extraction module.
    _ftr_ext = {'choices': ['cnn', 'bert', 'lstm_lm', 'lstm']}
    num_units: int = 128  # word embedding size.
    num_units_for_id_ftr: int = 128  # id feature embedding size.
    sp_emb_size: int = 1  # Embedding size of sparse features
    num_hidden: List[int] = 0  # hidden size.
    num_wide: int = 0  # number of wide features per doc.
    num_wide_sp: int = None  # number of sparse wide features per doc
    use_deep: bool = True  # Whether to use deep features.
    _use_deep = {'type': lambda s: True if s.lower() == 'true' else False if s.lower() == 'false' else s}
    elem_rescale: bool = True  # Whether to perform elementwise rescaling.
    use_doc_projection: bool = False  # whether to project multiple doc features to 1 vector.
    use_usr_projection: bool = False  # whether to project multiple usr features to 1 vector.

    # Ranking specific
    ltr_loss_fn: str = 'pairwise'  # learning-to-rank method.
    emb_sim_func: List[str] = ['inner']  # Approach to computing query/doc similarity scores
    _emb_sim_func = {"choices": ('inner', 'hadamard', 'concat', 'diff')}

    # Classification specific
    num_classes: int = 1  # Number of classes for multi-class classification tasks.

    # CNN related
    filter_window_sizes: List[int] = 3  # CNN filter window sizes.
    num_filters: int = 100  # number of CNN filters.
    explicit_empty: bool = False  # Explicitly modeling empty string in cnn

    # BERT related
    lr_bert: float = None  # Learning rate factor for bert components
    bert_config_file: str = None  # bert config.
    bert_checkpoint: str = None  # pretrained bert model checkpoint.

    # LSTM related
    unit_type: str = 'lstm'  # RNN cell unit type. Currently only supports lstm. Will support other cell types in the future
    _unit_type = {'choices': ['lstm']}
    num_layers: int = 1  # RNN layers
    num_residual_layers: int = 0  # Number of residual layers from top to bottom. For example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN cells in the returned list will be wrapped with `ResidualWrapper`. # noqa: E501
    forget_bias: float = 1.  # Forget bias of RNN cell
    rnn_dropout: float = 0.  # Dropout of RNN cell
    bidirectional: bool = False  # Whether to use bidirectional RNN
    normalized_lm: bool = False  # Whether to use normalized lm. This option only works for unit_type=lstm_lm

    # Optimizer
    optimizer: str = 'sgd'  # Type of optimizer to use. bert_adam is similar to the optimizer implementation in bert.
    _optimizer = {'choices': ['sgd', 'adam', 'bert_adam', 'bert_lamb']}
    max_gradient_norm: float = 1.0  # Clip gradients to this norm.
    learning_rate: float = 1.0  # Learning rate. Adam: 0.001 | 0.0001
    num_train_steps: int = 1  # Num steps to train.
    num_epochs: int = None  # Num of epochs to train, will overwrite train_steps if set
    num_warmup_steps: int = 0  # Num steps for warmup.
    train_batch_size: int = 32  # Training data batch size.
    test_batch_size: int = 32  # Test data batch size.
    l1: float = None  # Scale of L1 regularization
    l2: float = None  # Scale of L2 regularization

    # Data
    train_file: str = None  # Train file.
    dev_file: str = None  # Dev file.
    test_file: str = None  # Test file.
    out_dir: str = None  # Store log/model files.
    std_file: str = None  # feature standardization file
    max_len: int = 32  # max sent length.
    min_len: int = 3  # min sent length.

    # Vocab and word embedding
    vocab_file: str = None  # Vocab file
    we_file: str = None  # Pretrained word embedding file
    we_trainable: bool = True  # Whether to train word embedding
    PAD: str = '[PAD]'  # Token for padding
    SEP: str = '[SEP]'  # Token for sentence separation
    CLS: str = '[CLS]'  # Token for start of sentence
    UNK: str = '[UNK]'  # Token for unknown word
    MASK: str = '[MASK]'  # Token for masked word

    # Vocab and word embedding for id features
    vocab_file_for_id_ftr: str = None  # Vocab file for id features
    we_file_for_id_ftr: str = None  # Pretrained word embedding file for id features
    we_trainable_for_id_ftr: bool = True  # Whether to train word embedding for id features
    PAD_FOR_ID_FTR: str = '[PAD]'  # Padding token for id features
    UNK_FOR_ID_FTR: str = '[UNK]'  # Unknown word token for id features

    # Misc
    random_seed: int = 1234  # Random seed (>0, set a specific seed).
    steps_per_stats: int = 100  # training steps to print statistics.
    num_eval_rounds: int = None  # number of evaluation round, this param will override steps_per_eval as max(1, num_train_steps / num_eval_rounds)
    steps_per_eval: int = 1000  # training steps to evaluate datasets.
    keep_checkpoint_max: int = 5  # The maximum number of recent checkpoint files to keep. If 0, all checkpoint files are kept. Defaults to 5
    feature_names: List[str] = None  # the feature names.
    _feature_names = __comma_split_list(str)
    lambda_metric: str = None  # only support ndcg.
    init_weight: float = 0.1  # weight initialization value.
    pmetric: str = None  # Primary metric.
    all_metrics: List[str] = None  # All metrics.
    score_rescale: List[float] = None  # The mean and std of previous model. For score rescaling, the score_rescale has the xgboost mean and std.

    tokenization: str = 'punct'  # The tokenzation performed for data preprocessing. Currently support: punct/plain(no split). Note that this should be set correctly to ensure consistency for savedmodel.# noqa: E501
    _tokenization = {'choices': ['plain', 'punct']}

    resume_training: bool = False  # Whether to resume training from checkpoint in out_dir.
    metadata_path: str = None  # The metadata_path for converted avro2tf avro data.

    # tf-ranking related
    use_tfr_loss: bool = False  # whether to use tf-ranking loss.
    tfr_loss_fn: str = tfr.losses.RankingLossKey.SOFTMAX_LOSS  # tf-ranking loss
    _tfr_loss_fn = {'choices': [tfr.losses.RankingLossKey.SOFTMAX_LOSS, tfr.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS]}
    tfr_lambda_weights: str = None  #

    use_horovod: bool = False  # whether to use horovod for sync distributed training

    # multitask training related
    task_ids: List[int] = None  # All types of task IDs for multitask training. E.g., 1,2,3
    _task_ids = __comma_split_list(int)
    task_weights: List[float] = None  # Weights for each task specified in task_ids. E.g., 0.5,0.3,0.2
    _task_weights = __comma_split_list(float)


def get_hparams():
    """
    Get hyper-parameters.
    """
    hparams = tf.contrib.training.HParams(**Args()._asdict())
    # Print all hyper-parameters
    for k, v in sorted(vars(hparams).items()):
        print('--' + k + '=' + str(v))

    return hparams


def main(_):
    """
    This is the main method for training the model.
    :param argv: training parameters
    :return:
    """
    # Get executor task type from TF_CONFIG
    task_type = executor_utils.get_executor_task_type()

    # Get hyper-parameters.
    hparams = get_hparams()

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

    # For data sharding when using horovod
    hparams.add_hparam("hvd_info", None)
    if hparams.use_horovod:
        import horovod.tensorflow as hvd
        hvd.init()
        hparams.num_train_steps = hparams.num_train_steps // hvd.size()
        hparams.num_warmup_steps = hparams.num_warmup_steps // hvd.size()
        hparams.steps_per_eval = hparams.steps_per_eval // hvd.size()
        hparams.steps_per_stats = hparams.steps_per_stats // hvd.size()
        hparams.hvd_info = {'rank': hvd.rank(), 'size': hvd.size()}

    # Create directory and launch tensorboard
    if ((task_type == executor_utils.CHIEF or task_type == executor_utils.LOCAL_MODE) and (
            not hparams.use_horovod or (hparams.use_horovod and hvd.rank() == 0))):
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

    if ((task_type == executor_utils.EVALUATOR or task_type == executor_utils.LOCAL_MODE) and (
            not hparams.use_horovod or (hparams.use_horovod and hvd.rank() == 0))):
        # set up logger for evaluator
        sys.stdout = logger.Logger(os.path.join(hparams.out_dir, 'eval_log.txt'))

    hparams = misc_utils.extend_hparams(hparams)

    logging.info("***********DeText Training***********")

    # Train and evaluate DeText model
    train.train(hparams, input_fn)


if __name__ == '__main__':
    tf.compat.v1.app.run()
