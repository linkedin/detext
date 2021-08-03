from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import tensorflow_ranking as tfr
from absl import logging
from smart_arg import LateInit

from detext.utils import parsing_utils
from detext.utils.parsing_utils import InputFtrType, TaskType


class Arg:
    """Helper class for cooperative multi-inheritance"""

    def _set_late_init_attr(self, attr, value):
        """Sets an attribute as empty list if it's LateInit"""
        if getattr(self, attr) is LateInit:
            setattr(self, attr, value)

    def __post_init__(self):
        pass


@dataclass
class DatasetArg(Arg):
    """Dataset related arguments"""
    distribution_strategy: str = ''  # Distributed training strategy. Reference: tf official models: official/common/distribute_utils.py#L102
    __distribution_strategy = {'choices': ['one_device', 'mirrored', 'parameter_server', 'multi_worker_mirrored', 'tpu']}
    all_reduce_alg: Optional[str] = None  # All reduce algorithm. Reference: tf official models: official/common/distribute_utils.py#L102
    __all_reduce_alg = {'choices': ["hierarchical_copy", "nccl", "ring"]}
    num_gpu: int = -1  # Number of GPU for training

    run_eagerly: bool = False  # Whether to run in eager mode. Use True for debugging and False for speed

    train_file: str = ''  # Train file.
    dev_file: str = ''  # Dev file.
    test_file: str = ''  # Test file.
    out_dir: str = ''  # Store log/model files.

    num_train_steps: int = 1  # Num steps to train.
    num_eval_steps: int = 0  # Num steps to eval.
    num_epochs: int = 0  # Num of epochs to train, will overwrite train_steps if set
    steps_per_stats: int = 100  # training steps to print statistics.
    num_eval_rounds: int = 0  # number of evaluation round, this param will override steps_per_eval as max(1, num_train_steps / num_eval_rounds)
    steps_per_eval: int = 1000  # training steps to evaluate datasets.

    resume_training: bool = False  # Whether to resume training from checkpoint in out_dir.
    keep_checkpoint_max: int = 5  # The maximum number of recent checkpoint files to keep. If 0, all checkpoint files are kept. Defaults to 5

    train_batch_size: int = 32  # Training data batch size.
    test_batch_size: int = 32  # Test data batch size.

    def __post_init__(self):
        super().__post_init__()

        # Check matches between distribution strategy and all reduce algorithm. Reference: tf official models: official/common/distribute_utils.py#L102
        if self.all_reduce_alg is not None:
            if self.distribution_strategy == 'mirrored':
                assert self.all_reduce_alg in [None, 'nccl', 'hierarchical_copy']
            elif self.distribution_strategy == 'multi_worker_mirrored':
                assert self.all_reduce_alg in [None, 'nccl', 'ring']
            else:
                raise NotImplementedError(
                    f'Unsupported all reduce algorithm {self.all_reduce_alg} for chosen distribution strategy {self.distribution_strategy}')

        assert self.train_file, 'training_file must be specified'
        assert self.dev_file, 'dev_file must be specified'
        assert self.test_file, 'test_file must be specified'

        assert self.out_dir, 'out_dir must be specified'

        assert self.num_gpu >= 0, 'num_gpu must be specified'
        assert self.distribution_strategy, 'distribution_strategy must be specified'
        assert self.keep_checkpoint_max >= 0, 'keep_checkpoint_max must >= 0'

        # If epoch is set, overwrite training steps
        if self.num_epochs:
            steps_per_epoch = parsing_utils.estimate_steps_per_epoch(self.train_file, self.train_batch_size)
            self.num_train_steps = steps_per_epoch * self.num_epochs

        # If num_eval_rounds is set, override steps_per_eval
        assert self.num_eval_rounds >= 0, 'num_eval_rounds must be non-negative integers'
        if self.num_eval_rounds:
            self.steps_per_eval = max(1, int(self.num_train_steps / self.num_eval_rounds))

        if self.steps_per_stats > self.steps_per_eval:
            logging.error('steps_per_stats: %d is specified to be greater than steps_per_eval: %d, we will use steps_per_eval as'
                          ' steps_per_stats.', self.steps_per_stats, self.steps_per_eval)
            self.steps_per_stats = self.steps_per_eval


@dataclass
class FeatureArg(Arg):
    """Feature related arguments"""
    query_column_name: str = ''  # Name of the query field
    label_column_name: str = ''  # Name of the label field
    weight_column_name: str = ''  # Name of the weight field
    uid_column_name: str = ''  # Name of the uid field

    task_id_column_name: str = ''  # Name of the task id field
    task_ids: Optional[List[int]] = None  # All types of task IDs for multitask training. E.g., 1,2,3
    task_weights: Optional[List[float]] = None  # Weights for each task specified in task_ids. E.g., 0.5,0.3,0.2

    dense_ftrs_column_names: List[str] = LateInit  # Names of dense features columns
    nums_dense_ftrs: List[int] = LateInit  # Numbers of dense features per document for each dense features column. nums_dense_ftrs[i] = shape(dense_ftrs[i])

    sparse_ftrs_column_names: List[str] = LateInit  # Names of sparse features columns
    nums_sparse_ftrs: List[int] = LateInit  # Cardinality of each sparse features nums_sparse_ftrs[i] = cardinality(sparse_ftrs[i])

    user_text_column_names: List[str] = LateInit  # Names of the user text fields
    doc_text_column_names: List[str] = LateInit  # Names of the document text fields
    user_id_column_names: List[str] = LateInit  # Names of the user id fields
    doc_id_column_names: List[str] = LateInit  # Names of the document id fields

    std_file: Optional[str] = None  # feature standardization file

    # Vocab and word embedding
    vocab_file: str = ''  # Vocab file
    vocab_hub_url: str = ''  # TF hub url to vocab layer
    we_file: str = ''  # Pretrained word embedding file
    embedding_hub_url: str = ''  # TF hub url to embedding layer
    we_trainable: bool = True  # Whether to train word embedding
    PAD: str = '[PAD]'  # Token for padding
    SEP: str = '[SEP]'  # Token for sentence separation
    CLS: str = '[CLS]'  # Token for start of sentence
    UNK: str = '[UNK]'  # Token for unknown word
    MASK: str = '[MASK]'  # Token for masked word

    # Vocab and word embedding for id features
    vocab_file_for_id_ftr: str = ''  # Vocab file for id features
    vocab_hub_url_for_id_ftr: str = ''  # TF hub url to vocab layer for id feature
    we_file_for_id_ftr: str = ''  # Pretrained word embedding file for id features
    embedding_hub_url_for_id_ftr: str = ''  # TF hub url to embedding layer for id features
    we_trainable_for_id_ftr: bool = True  # Whether to train word embedding for id features
    PAD_FOR_ID_FTR: str = '[PAD]'  # Padding token for id features
    UNK_FOR_ID_FTR: str = '[UNK]'  # Unknown word token for id features

    max_len: int = 32  # Max sent length.
    min_len: int = 3  # Min sent length.

    # Late init variables inferred from post initialization. DO NOT pass any values to the following arguments
    feature_type2name: Dict[str, str] = LateInit  # Late init only. DO NOT pass value. Map of feature type to feature names

    has_query: bool = LateInit  # Late init only. DO NOT pass value. Whether there's query in feature names
    use_dense_ftrs: bool = LateInit  # Late init only. DO NOT pass value. Indicator of whether there's dense_ftrs
    total_num_dense_ftrs: int = LateInit  # Late init only. DO NOT pass value. Total number of dense features
    use_sparse_ftrs: bool = LateInit  # Late init only. DO NOT pass value. Indicator of whether there's dense_ftrs
    total_num_sparse_ftrs: int = LateInit  # Late init only. DO NOT pass value. Total number of sparse features
    use_deep: bool = LateInit  # Late init only. DO NOT pass value. Whether to use deep features.

    num_doc_fields: int = LateInit  # Late init only. DO NOT pass value. Number of document text fields
    num_user_fields: int = LateInit  # Late init only. DO NOT pass value. Number of user text fields
    num_doc_id_fields: int = LateInit  # Late init only. DO NOT pass value. Number of document id fields
    num_user_id_fields: int = LateInit  # Late init only. DO NOT pass value. Number of user id fields
    num_id_fields: int = LateInit  # Late init only. DO NOT pass value. Number of id fields
    num_text_fields: int = LateInit  # Late init only. DO NOT pass value. Number of text fields

    ftr_mean: Optional[List[float]] = LateInit  # Late init only. DO NOT pass value. Mean of dense features
    ftr_std: Optional[List[float]] = LateInit  # Late init only. DO NOT pass value. Std of dense features

    def __post_init__(self):
        super().__post_init__()
        for attr in ['nums_dense_ftrs', 'nums_sparse_ftrs', 'user_id_column_names', 'sparse_ftrs_column_names', 'dense_ftrs_column_names',
                     'user_text_column_names', 'doc_id_column_names', 'doc_text_column_names']:
            self._set_late_init_attr(attr, [])

        # Assemble feature map
        self.feature_type2name = dict()
        all_ftr_names = list()
        for ftr_type in parsing_utils.get_feature_types():
            assert hasattr(self, ftr_type), f'{ftr_type} must be defined in DeText argument parser'
            ftr_name = getattr(self, ftr_type)
            if ftr_name:
                self.feature_type2name[ftr_type] = ftr_name

                ftr_name = [ftr_name] if not isinstance(ftr_name, list) else ftr_name
                all_ftr_names += ftr_name
                assert len(set(all_ftr_names)) == len(
                    all_ftr_names), f'Duplicate feature names for feature type {ftr_type}'

        # Multi-task training: currently only support ranking tasks with both deep and dense features
        if self.task_ids:
            assert InputFtrType.TASK_ID_COLUMN_NAME in self.feature_type2name, "task_id feature not found for multi-task training"

            # Parse task ids an weights from inputs and convert them into a map
            task_ids = self.task_ids
            raw_weights = self.task_weights if self.task_weights else [1.0] * len(task_ids)
            task_weights = [float(wt) / sum(raw_weights) for wt in raw_weights]  # Normalize task weights

            # Check size of task_ids and task_weights
            assert len(task_ids) == len(task_weights), "size of task IDs and weights must match"

            self.task_weights = task_weights

        # Calculate total number of dense features
        assert len(self.nums_dense_ftrs) == len(self.dense_ftrs_column_names), \
            f'Number of dense features must be consistent in shape and name. E.g., if there are k features in --dense_ftrs, you will also need to specify ' \
            f'k integers for --nums_dense_ftrs. Current setting: len(nums_dense_ftrs) = {len(self.nums_dense_ftrs)}, ' \
            f'len(dense_ftrs) = {len(self.dense_ftrs_column_names)}'
        self.total_num_dense_ftrs = 0
        for num_dense_ftrs in self.nums_dense_ftrs:
            self.total_num_dense_ftrs += num_dense_ftrs

        self.total_num_sparse_ftrs = 0
        for num_sparse_ftrs in self.nums_sparse_ftrs:
            self.total_num_sparse_ftrs += num_sparse_ftrs

        # Get text field numbers
        self.has_query = InputFtrType.QUERY_COLUMN_NAME in self.feature_type2name
        self.num_doc_fields = len(self.feature_type2name.get(InputFtrType.DOC_TEXT_COLUMN_NAMES, []))
        self.num_user_fields = len(self.feature_type2name.get(InputFtrType.USER_TEXT_COLUMN_NAMES, []))
        self.num_text_fields = self.has_query + self.num_doc_fields + self.num_user_fields

        if self.ftr_ext != 'bert' and (self.num_doc_fields > 0 or self.num_user_fields > 0 or self.has_query):
            assert self.vocab_file or self.vocab_hub_url or self.embedding_hub_url, \
                "Must provide vocab or embedding file when text features are provided when not using bert"

        # Get id field numbers
        self.num_doc_id_fields = len(self.feature_type2name.get(InputFtrType.DOC_ID_COLUMN_NAMES, []))
        self.num_user_id_fields = len(self.feature_type2name.get(InputFtrType.USER_ID_COLUMN_NAMES, []))
        self.num_id_fields = self.num_doc_id_fields + self.num_user_id_fields
        if self.num_doc_id_fields > 0 or self.num_user_id_fields > 0:
            assert self.vocab_file_for_id_ftr or self.vocab_hub_url_for_id_ftr or self.embedding_hub_url_for_id_ftr, \
                "Must provide vocab or embedding file for id features arg when id features are provided"

        assert self.use_deep is LateInit, "use_deep is an inferred argument. Do not pass values to this argument"
        self.use_deep = any([self.num_text_fields, self.num_id_fields])

        # Get indicator of whether dense/sparse features are used
        self.use_dense_ftrs = InputFtrType.DENSE_FTRS_COLUMN_NAMES in self.feature_type2name
        self.use_sparse_ftrs = InputFtrType.SPARSE_FTRS_COLUMN_NAMES in self.feature_type2name

        # Feature normalization
        if self.std_file:
            # Read normalization file
            ftr_mean, ftr_std = parsing_utils.load_ftr_mean_std(self.std_file)
            self.ftr_mean = np.array(ftr_mean, dtype=np.float32).tolist()
            self.ftr_std = np.array(ftr_std, dtype=np.float32).tolist()
        else:
            self.ftr_mean = None
            self.ftr_std = None


@dataclass
class NetworkArg(Arg):
    """Network related arguments"""
    random_seed: int = 1234  # Random seed (>0, set a specific seed).

    ftr_ext: str = 'cnn'  # NLP feature extraction module.
    __ftr_ext = {'choices': ['cnn', 'bert', 'lstm']}
    num_units: int = 128  # word embedding size.
    num_units_for_id_ftr: int = 128  # id feature embedding size.

    sparse_embedding_size: int = 1  # Embedding size of sparse features
    sparse_embedding_cross_ftr_combiner: str = 'sum'  # How to combine sparse embeddings for different features. If set to 'concat', embeddings of different features will be concatenated. E.g. given company id embedding (dim=10) and user id embedding (dim=10), they will be concatenated to embedding with dim=20. 'sum' means summation (resulting embedding dim=10) # noqa: E501
    __sparse_embedding_cross_ftr_combiner = {'choices': ('sum', 'concat')}
    sparse_embedding_same_ftr_combiner: str = 'sum'  # How to combine sparse embeddings for same feature.  If set to 'sum', embeddings of the same feature will be summed at each dim. E.g., given company ids 100 and 23, whose embedding is [0.5, 1.0] and [1.1, 2.3], the summed embedding will be [1.6(0.5+1.1), 3.3(1.0 + 2.3)]. 'mean' means taking average of the embeddings (resulting embedding = [0.8, 1.65]). # noqa: E501
    __sparse_embedding_same_ftr_combiner = {'choices': ('sum', 'mean')}
    num_hidden: List[int] = LateInit  # hidden size.

    rescale_dense_ftrs: bool = True  # Whether to perform elementwise rescaling on dense_ftrs.

    add_doc_projection: bool = False  # whether to project multiple doc features to 1 vector.
    add_user_projection: bool = False  # whether to project multiple user features to 1 vector.

    emb_sim_func: List[str] = LateInit  # Approach to computing query/doc similarity scores
    __emb_sim_func = {"choices": ('inner', 'hadamard', 'concat', 'diff')}

    # CNN
    filter_window_sizes: List[int] = LateInit  # CNN filter window sizes.
    num_filters: int = 100  # number of CNN filters.

    # BERT related
    lr_bert: float = 0.0  # Learning rate factor for bert components
    bert_hub_url: Optional[str] = None  # bert tf-hub model url.

    # LSTM related
    num_layers: int = 1  # RNN layers
    forget_bias: float = 1.  # Forget bias of RNN cell
    rnn_dropout: float = 0.  # Dropout of RNN cell
    bidirectional: bool = False  # Whether to use bidirectional RNN

    # Late init variables inferred from post initialization. DO NOT pass any values to the following arguments
    max_filter_window_size: int = LateInit  # Late init only. DO NOT pass value. Maximum CNN filter windows size

    def __post_init__(self):
        super().__post_init__()

        self._set_late_init_attr('num_hidden', [])
        self._set_late_init_attr('emb_sim_func', ['inner'])
        self._set_late_init_attr('filter_window_sizes', [3])

        # If not using cnn models, then disable cnn parameters
        if self.ftr_ext != 'cnn':
            self.filter_window_sizes = [0]
        self.max_filter_window_size = max(self.filter_window_sizes)

        # If using bert, the bert_hub_url must be provided
        if self.ftr_ext == 'bert':
            assert self.bert_hub_url is not None, "When using bert as ftr_ext, bert_hub_url must be specified."


@dataclass
class OptimizationArg(Arg):
    """Optimization related arguments"""
    use_lr_schedule: bool = True  # Whether to use warmup and decay on learning rate
    num_warmup_steps: int = 0  # Num steps for warmup. TODO: change to warm up ratio in the future
    optimizer: str = 'sgd'  # Type of optimizer to use. adamw is the AdamWeightDecay optimizer
    use_bias_correction_for_adamw: bool = False  # Whether to use bias correction for AdamWeightDecay optimizer
    __optimizer = {'choices': ['sgd', 'adam', 'adamw', 'lamb']}
    max_gradient_norm: float = 1.0  # Clip gradients to this norm.
    learning_rate: float = 1.0  # Learning rate. Adam: 0.001 | 0.0001

    task_type: str = TaskType.RANKING  # Task type. ranking/binary classification/classification(multiclass classification)
    __task_type = {'choices': [TaskType.RANKING, TaskType.CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION, TaskType.BINARY_CLASSIFICATION]}
    num_classes: int = 1  # Number of classes for multi-class classification tasks.

    l1: float = 0.  # Scale of L1 regularization
    l2: float = 0.  # Scale of L2 regularization

    pmetric: Optional[str] = None  # Primary metric.
    all_metrics: Optional[List[str]] = None  # All metrics.

    ltr_loss_fn: str = 'pairwise'  # learning-to-rank method.
    use_tfr_loss: bool = False  # whether to use tf-ranking loss.
    tfr_loss_fn: str = tfr.losses.RankingLossKey.SOFTMAX_LOSS  # tf-ranking loss
    __tfr_loss_fn = {'choices': [tfr.losses.RankingLossKey.SOFTMAX_LOSS, tfr.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS]}
    tfr_lambda_weights: Optional[str] = None  # TFR lambda weights

    explicit_allreduce: bool = True  # Whether to perform explicit allreduce

    lambda_metric: Optional[str] = None  # only support ndcg.

    def __post_init__(self):
        super().__post_init__()
        assert self.l1 >= 0, "l1 scale must be non-negative"
        assert self.l2 >= 0, "l2 scale must be non-negative"

        assert self.pmetric is not None, "Please set your primary evaluation metric using --pmetric option"
        assert self.pmetric != 'confusion_matrix', 'confusion_matrix cannot be used as primary evaluation metric.'

        # Set all relevant evaluation metrics
        all_metrics = self.all_metrics if self.all_metrics else [self.pmetric]
        assert self.pmetric in all_metrics, "pmetric must be within all_metrics"
        self.all_metrics = all_metrics

        # Lambda rank
        if self.lambda_metric is not None and self.lambda_metric == 'ndcg':
            self.lambda_metric = {'metric': 'ndcg', 'topk': 10}
        else:
            self.lambda_metric = None

        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.task_type = TaskType.CLASSIFICATION  # Use same impl of classification

        if self.task_type == TaskType.CLASSIFICATION:
            assert self.num_classes > 1, '`num_classes must be larger than 1 for classification/multiclass classification `task_type`.'
            assert self.pmetric == 'accuracy', 'Primary metric (`pmetric`) must be set to \'accuracy\' ' \
                                               'for classification/multiclass classification.'

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            assert self.num_classes in [1, 2], '`num_classes must be set to 1 or 2 (either is ok) for binary_classification `task_type`.'
            self.num_classes = 1

            supported_metrics = {'accuracy', 'auc'}
            for metric in self.all_metrics:
                assert metric in supported_metrics, f"Metric {metric} not supported. Must be one of {supported_metrics}"
