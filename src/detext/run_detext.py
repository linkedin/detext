"""
Overall pipeline to train the model.  It parses arguments, and trains a DeText model.
"""

import logging
import os
import sys
import time
from typing import NamedTuple, List, Optional, Dict

import tensorflow as tf
import tensorflow_ranking as tfr
from detext.train import train
from detext.train.data_fn import input_fn
from detext.utils import misc_utils, logger
from detext.utils.executor_utils import EVALUATOR, LOCAL_MODE, CHIEF, get_executor_task_type
from linkedin.smart_arg import arg_suite
from tensorflow.contrib.training import HParams


@arg_suite
class DetextArg(NamedTuple):
    """
    DeText: a Deep Text understanding framework for NLP related ranking, classification, and language generation tasks.
    It leverages semantic matching using deep neural networks to
    understand member intents in search and recommender systems.
    As a general NLP framework, currently DeText can be applied to many tasks,
    including search & recommendation ranking, multi-class classification and query understanding tasks.
    """
    #  kwargs utility function for split a str to a List, provided for backward compatibilities
    #  Please use built-in List parsing support when adding new arguments
    __comma_split_list = lambda t: {'type': lambda s: [t(item) for item in s.split(',')] if ',' in s else t(s), 'nargs': None}  # noqa: E731
    # network
    ftr_ext: str  # NLP feature extraction module.
    _ftr_ext = {'choices': ['cnn', 'bert', 'lstm_lm', 'lstm']}
    num_units: int = 128  # word embedding size.
    num_units_for_id_ftr: int = 128  # id feature embedding size.
    sp_emb_size: int = 1  # Embedding size of sparse features
    num_hidden: List[int] = [0]  # hidden size.
    num_wide: int = 0  # number of wide features per doc.
    num_wide_sp: Optional[int] = None  # number of sparse wide features per doc
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
    filter_window_sizes: List[int] = [3]  # CNN filter window sizes.
    num_filters: int = 100  # number of CNN filters.
    explicit_empty: bool = False  # Explicitly modeling empty string in cnn

    # BERT related
    lr_bert: Optional[float] = None  # Learning rate factor for bert components
    bert_config_file: Optional[str] = None  # bert config.
    bert_checkpoint: Optional[str] = None  # pretrained bert model checkpoint.
    use_bert_dropout: bool = False  # apply dropout during training in bert layers.

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
    num_epochs: Optional[int] = None  # Num of epochs to train, will overwrite train_steps if set
    num_warmup_steps: int = 0  # Num steps for warmup.
    train_batch_size: int = 32  # Training data batch size.
    test_batch_size: int = 32  # Test data batch size.
    l1: Optional[float] = None  # Scale of L1 regularization
    l2: Optional[float] = None  # Scale of L2 regularization

    # Data
    train_file: Optional[str] = None  # Train file.
    dev_file: Optional[str] = None  # Dev file.
    test_file: Optional[str] = None  # Test file.
    out_dir: Optional[str] = None  # Store log/model files.
    std_file: Optional[str] = None  # feature standardization file
    max_len: int = 32  # max sent length.
    min_len: int = 3  # min sent length.

    # Vocab and word embedding
    vocab_file: Optional[str] = None  # Vocab file
    we_file: Optional[str] = None  # Pretrained word embedding file
    we_trainable: bool = True  # Whether to train word embedding
    PAD: str = '[PAD]'  # Token for padding
    SEP: str = '[SEP]'  # Token for sentence separation
    CLS: str = '[CLS]'  # Token for start of sentence
    UNK: str = '[UNK]'  # Token for unknown word
    MASK: str = '[MASK]'  # Token for masked word

    # Vocab and word embedding for id features
    vocab_file_for_id_ftr: Optional[str] = None  # Vocab file for id features
    we_file_for_id_ftr: Optional[str] = None  # Pretrained word embedding file for id features
    we_trainable_for_id_ftr: bool = True  # Whether to train word embedding for id features
    PAD_FOR_ID_FTR: str = '[PAD]'  # Padding token for id features
    UNK_FOR_ID_FTR: str = '[UNK]'  # Unknown word token for id features

    # Misc
    random_seed: int = 1234  # Random seed (>0, set a specific seed).
    steps_per_stats: int = 100  # training steps to print statistics.
    num_eval_rounds: Optional[int] = None  # number of evaluation round, this param will override steps_per_eval as max(1, num_train_steps / num_eval_rounds)
    steps_per_eval: int = 1000  # training steps to evaluate datasets.
    keep_checkpoint_max: int = 5  # The maximum number of recent checkpoint files to keep. If 0, all checkpoint files are kept. Defaults to 5
    feature_names: Optional[List[str]] = None  # the feature names.
    _feature_names = __comma_split_list(str)
    lambda_metric: Optional[str] = None  # only support ndcg.
    init_weight: float = 0.1  # weight initialization value.
    pmetric: Optional[str] = None  # Primary metric.
    all_metrics: Optional[List[str]] = None  # All metrics.
    score_rescale: Optional[List[float]] = None  # The mean and std of previous model. For score rescaling, the score_rescale has the xgboost mean and std.

    add_first_dim_for_query_placeholder: bool = False  # Whether to add a batch dimension for query and usr_* placeholders. This shall be set to True if the query field is used as document feature in model serving.# noqa: E501
    add_first_dim_for_usr_placeholder: bool = False  # Whether to add a batch dimension for query and usr_* placeholders. This shall be set to True if usr fields are used document feature in model serving.# noqa: E501

    tokenization: str = 'punct'  # The tokenzation performed for data preprocessing. Currently support: punct/plain(no split). Note that this should be set correctly to ensure consistency for savedmodel.# noqa: E501
    _tokenization = {'choices': ['plain', 'punct']}

    resume_training: bool = False  # Whether to resume training from checkpoint in out_dir.
    metadata_path: Optional[str] = None  # The metadata_path for converted avro2tf avro data.

    # tf-ranking related
    use_tfr_loss: bool = False  # whether to use tf-ranking loss.
    tfr_loss_fn: str = tfr.losses.RankingLossKey.SOFTMAX_LOSS  # tf-ranking loss
    _tfr_loss_fn = {'choices': [tfr.losses.RankingLossKey.SOFTMAX_LOSS, tfr.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS]}
    tfr_lambda_weights: Optional[str] = None  #

    use_horovod: bool = False  # whether to use horovod for sync distributed training
    hvd_info: Optional[Dict[str, int]] = None
    # multitask training related
    task_ids: Optional[List[int]] = None  # All types of task IDs for multitask training. E.g., 1,2,3
    _task_ids = __comma_split_list(int)
    task_weights: Optional[List[float]] = None  # Weights for each task specified in task_ids. E.g., 0.5,0.3,0.2
    _task_weights = __comma_split_list(float)

    # This method is automatically called by smart-arg once the argument is created by parsing cli or the constructor
    # It's used to late-initialize some fields after other fields are created.
    def __late_init__(self):
        arg = self
        # if epoch is set, overwrite training steps
        if arg.num_epochs is not None:
            arg = arg._replace(num_train_steps=misc_utils.estimate_train_steps(
                arg.train_file,
                arg.num_epochs,
                arg.train_batch_size,
                arg.metadata_path is None))

        # if num_eval_rounds is set, override steps_per_eval
        if arg.num_eval_rounds is not None:
            arg = arg._replace(steps_per_eval=max(1, int(arg.num_train_steps / arg.num_eval_rounds)))

        # For data sharding when using horovod
        if arg.use_horovod:
            import horovod.tensorflow as hvd
            hvd.init()
            arg = arg._replace(
                num_train_steps=arg.num_train_steps // hvd.size(),
                num_warmup_steps=arg.num_warmup_steps // hvd.size(),
                steps_per_eval=arg.steps_per_eval // hvd.size(),
                steps_per_stats=arg.steps_per_stats // hvd.size(),
                hvd_info={'rank': hvd.rank(), 'size': hvd.size()})

        return arg


def get_hparams(argument: DetextArg = None):
    """
    Get hyper-parameters.
    Not in use and to be removed.
    """
    import warnings
    warnings.warn("get_hparams is deprecated and not used, call `run_detext` directly")
    hparams = HParams(**(argument or DetextArg.__from_argv__(error_on_unknown=False))._asdict())
    # Print all hyper-parameters
    for k, v in sorted(vars(hparams).items()):
        logging.info(f'{k} = {v}')

    return hparams


def main(argv):
    """
    This is the main method for training the model.
    :param argv: training parameters
    """

    argument = DetextArg.__from_argv__(argv[1:], error_on_unknown=False)
    run_detext(argument)


def run_detext(argument: DetextArg):
    logging.info(f"Args:\n {argument}")

    tf.logging.set_verbosity(tf.logging.INFO)

    # For data sharding when using horovod
    if argument.use_horovod:
        import horovod.tensorflow as hvd
    else:
        hvd = None

    # Get executor task type from TF_CONFIG
    task_type = get_executor_task_type()

    # Create directory and launch tensorboard
    master = not hvd or hvd.rank() == 0
    if (task_type == CHIEF or task_type == LOCAL_MODE) and master:
        # If not resume training from checkpoints, delete output directory.
        if not argument.resume_training:
            logging.info("Removing previous output directory...")
            if tf.gfile.Exists(argument.out_dir):
                tf.gfile.DeleteRecursively(argument.out_dir)

        # If output directory deleted or does not exist, create the directory.
        if not tf.gfile.Exists(argument.out_dir):
            logging.info('Creating dirs recursively at: {0}'.format(argument.out_dir))
            tf.gfile.MakeDirs(argument.out_dir)

        misc_utils.save_hparams(argument.out_dir, HParams(**argument._asdict()))

    else:
        # TODO: move removal/creation to a hadoopShellJob st. it does not reside in distributed training code.
        logging.info("Waiting for chief to remove/create directories.")
        # Wait for dir created form chief
        time.sleep(10)

    if (task_type == EVALUATOR or task_type == LOCAL_MODE) and master:
        # set up logger for evaluator
        sys.stdout = logger.Logger(os.path.join(argument.out_dir, 'eval_log.txt'))

    hparams = misc_utils.extend_hparams(HParams(**argument._asdict()))

    logging.info("***********DeText Training***********")

    # Train and evaluate DeText model
    train.train(hparams, input_fn)


if __name__ == '__main__':
    tf.compat.v1.app.run()
