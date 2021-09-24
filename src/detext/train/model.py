from functools import partial

import tensorflow as tf
from detext.layers.feature_grouper import FeatureGrouper
from detext.layers.feature_name_type_converter import FeatureNameTypeConverter, DEEP_FTR_TYPES
from detext.layers.feature_normalizer import FeatureNormalizer
from detext.layers.feature_rescaler import FeatureRescaler
from detext.layers.interaction_layer import InteractionLayer
from detext.layers.output_transform_layer import OutputTransformLayer
from detext.layers.representation_layer import RepresentationLayer
from detext.layers.scoring_layer import ScoringLayer
from detext.layers.shallow_tower_layer import ShallowTowerLayer
from detext.layers.sparse_embedding_layer import SparseEmbeddingLayer
from detext.utils.parsing_utils import InputFtrType, TaskType, OutputFtrType, InternalFtrType, iterate_items_with_list_val, as_list, get_feature_nums

TASK_TYPE_TO_TRAINING_OUTPUT_NAME = {
    TaskType.CLASSIFICATION: OutputFtrType.DETEXT_CLS_LOGITS,
    TaskType.BINARY_CLASSIFICATION: OutputFtrType.DETEXT_CLS_LOGITS,
    TaskType.RANKING: OutputFtrType.DETEXT_RANKING_SCORES
}

FTR_TYPE_TO_KERAS_INPUT_RANK = {
    InputFtrType.QUERY_COLUMN_NAME: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name),

    InputFtrType.DENSE_FTRS_COLUMN_NAMES: lambda name, num_dense_ftrs: tf.keras.Input(shape=(None, num_dense_ftrs), dtype='float32', name=name),
    InputFtrType.SPARSE_FTRS_COLUMN_NAMES: lambda name, num_sparse_ftrs: tf.keras.Input(shape=(None, num_sparse_ftrs), sparse=True, dtype='float32', name=name),
    InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES: lambda name, num_sparse_ftrs: tf.keras.Input(shape=(None, num_sparse_ftrs), sparse=True,
                                                                                                      dtype='float32', name=name),

    InputFtrType.TASK_ID_COLUMN_NAME: lambda name, *args: tf.keras.Input(shape=(), dtype='int64', name=name),
    InputFtrType.DOC_TEXT_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(None,), dtype='string', name=name),
    InputFtrType.DOC_ID_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(None,), dtype='string', name=name),
    InputFtrType.USER_TEXT_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name),
    InputFtrType.USER_ID_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name)
}

FTR_TYPE_TO_SCORER_TENSOR_SPEC_RANKING = {
    InternalFtrType.DOC_FTRS: lambda num_doc_fields, num_deep_ftrs: tf.TensorSpec(shape=(None, None, num_doc_fields, num_deep_ftrs), dtype='float32',
                                                                                  name=InternalFtrType.DOC_FTRS),
    InternalFtrType.USER_FTRS: lambda num_user_fields, num_deep_ftrs: tf.TensorSpec(shape=(None, num_user_fields, num_deep_ftrs), dtype='float32',
                                                                                    name=InternalFtrType.USER_FTRS),
    InternalFtrType.QUERY_FTRS: lambda num_deep_ftrs: tf.TensorSpec(shape=(None, num_deep_ftrs), dtype='float32',
                                                                    name=InternalFtrType.QUERY_FTRS),
    InputFtrType.TASK_ID_COLUMN_NAME: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='int64', name=name),
    InputFtrType.DENSE_FTRS_COLUMN_NAMES: lambda name, num_dense_ftrs: tf.TensorSpec(shape=(None, None, num_dense_ftrs), dtype='float32', name=name),
    InputFtrType.SPARSE_FTRS_COLUMN_NAMES: lambda num_sparse_ftrs: tf.SparseTensorSpec(shape=(None, None, num_sparse_ftrs), dtype='float32'),
    InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES: lambda num_sparse_ftrs: tf.SparseTensorSpec(shape=(None, None, num_sparse_ftrs), dtype='float32')
}

FTR_TYPE_TO_SCORER_TENSOR_SPEC_CLASSIFICATION = {
    InternalFtrType.DOC_FTRS: lambda num_doc_fields, num_deep_ftrs: tf.TensorSpec(shape=(None, None, num_doc_fields, num_deep_ftrs), dtype='float32',
                                                                                  name=InternalFtrType.DOC_FTRS),
    InternalFtrType.USER_FTRS: lambda num_user_fields, num_deep_ftrs: tf.TensorSpec(shape=(None, num_user_fields, num_deep_ftrs), dtype='float32',
                                                                                    name=InternalFtrType.USER_FTRS),
    InternalFtrType.QUERY_FTRS: lambda num_deep_ftrs: tf.TensorSpec(shape=(None, num_deep_ftrs), dtype='float32',
                                                                    name=InternalFtrType.QUERY_FTRS),
    InputFtrType.TASK_ID_COLUMN_NAME: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='int64', name=name),
    InputFtrType.DENSE_FTRS_COLUMN_NAMES: lambda name, num_dense_ftrs: tf.TensorSpec(shape=(None, num_dense_ftrs), dtype='float32', name=name),
    InputFtrType.SPARSE_FTRS_COLUMN_NAMES: lambda num_sparse_ftrs: tf.SparseTensorSpec(shape=(None, num_sparse_ftrs), dtype='float32'),
    InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES: lambda num_sparse_ftrs: tf.SparseTensorSpec(shape=(None, num_sparse_ftrs), dtype='float32'),
}

TASK_TYPE_TO_SCORER_TENSOR_SPEC = {
    TaskType.RANKING: FTR_TYPE_TO_SCORER_TENSOR_SPEC_RANKING,
    TaskType.BINARY_CLASSIFICATION: FTR_TYPE_TO_SCORER_TENSOR_SPEC_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION: FTR_TYPE_TO_SCORER_TENSOR_SPEC_CLASSIFICATION,
    TaskType.CLASSIFICATION: FTR_TYPE_TO_SCORER_TENSOR_SPEC_CLASSIFICATION
}

EMBEDDING_GENERATOR_TENSOR_SPEC_RANKING = {
    InputFtrType.QUERY_COLUMN_NAME: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name),
    InputFtrType.DOC_TEXT_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None, None), dtype='string', name=name),
    InputFtrType.DOC_ID_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None, None), dtype='string', name=name),
    InputFtrType.USER_TEXT_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name),
    InputFtrType.USER_ID_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name)
}

EMBEDDING_GENERATOR_TENSOR_SPEC_CLASSIFICATION = {
    InputFtrType.QUERY_COLUMN_NAME: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name),
    InputFtrType.DOC_TEXT_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name),
    InputFtrType.DOC_ID_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name),
    InputFtrType.USER_TEXT_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name),
    InputFtrType.USER_ID_COLUMN_NAMES: lambda name, *args: tf.TensorSpec(shape=(None,), dtype='string', name=name)
}

TASK_TYPE_TO_EMBEDDING_GENERATOR_TENSOR_SPEC = {
    TaskType.RANKING: EMBEDDING_GENERATOR_TENSOR_SPEC_RANKING,
    TaskType.BINARY_CLASSIFICATION: EMBEDDING_GENERATOR_TENSOR_SPEC_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION: EMBEDDING_GENERATOR_TENSOR_SPEC_CLASSIFICATION,
    TaskType.CLASSIFICATION: EMBEDDING_GENERATOR_TENSOR_SPEC_CLASSIFICATION
}

FTR_TYPE_TO_KERAS_INPUT_CLS = {
    InputFtrType.QUERY_COLUMN_NAME: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name),

    InputFtrType.DENSE_FTRS_COLUMN_NAMES: lambda name, num_dense_ftrs: tf.keras.Input(shape=(num_dense_ftrs,), dtype='float32', name=name),
    InputFtrType.SPARSE_FTRS_COLUMN_NAMES: lambda name, num_sparse_ftrs: tf.keras.Input(shape=(num_sparse_ftrs,), sparse=True, dtype='float32', name=name),
    InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES: lambda name, num_sparse_ftrs: tf.keras.Input(shape=(num_sparse_ftrs,), sparse=True, dtype='float32',
                                                                                                      name=name),

    InputFtrType.TASK_ID_COLUMN_NAME: lambda name, *args: tf.keras.Input(shape=(), dtype='int64', name=name),
    InputFtrType.DOC_TEXT_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name),
    InputFtrType.DOC_ID_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name),
    InputFtrType.USER_TEXT_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name),
    InputFtrType.USER_ID_COLUMN_NAMES: lambda name, *args: tf.keras.Input(shape=(), dtype='string', name=name)
}


def create_keras_inputs(feature_type2name, feature_name2num, task_type):
    """Returns a dictionary mapping feature names to input shape"""
    feature_dct = dict()
    input_schema = FTR_TYPE_TO_KERAS_INPUT_RANK if task_type == TaskType.RANKING else FTR_TYPE_TO_KERAS_INPUT_CLS
    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
        if ftr_type not in input_schema:
            continue
        if ftr_type == InputFtrType.DENSE_FTRS_COLUMN_NAMES:
            for ftr_name in ftr_name_lst:
                feature_dct[ftr_name] = input_schema[ftr_type](ftr_name, feature_name2num[ftr_name])
        elif ftr_type in [InputFtrType.SPARSE_FTRS_COLUMN_NAMES, InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES]:
            for ftr_name in ftr_name_lst:
                feature_dct[ftr_name] = input_schema[ftr_type](ftr_name, feature_name2num[ftr_name])
        else:
            for ftr_name in ftr_name_lst:
                feature_dct[ftr_name] = input_schema[ftr_type](ftr_name, None)

    return dict(sorted(feature_dct.items()))


def create_embedding_generator_signature(feature_type2name, task_type):
    """Returns the input signature of embedding generator"""
    input_signature = dict()
    embedding_generator_tensor_spec = TASK_TYPE_TO_EMBEDDING_GENERATOR_TENSOR_SPEC[task_type]
    for ftr_type, ftr_name_lst in iterate_items_with_list_val(feature_type2name):
        if ftr_type not in embedding_generator_tensor_spec:
            continue
        else:
            for ftr_name in ftr_name_lst:
                input_signature[ftr_name] = embedding_generator_tensor_spec[ftr_type](ftr_name)

    return [input_signature]


def create_scorer_input_signature(task_type, feature_type_to_name, feature_name_to_num, num_doc_fields, num_user_fields, num_deep_ftrs, has_query):
    """Returns the input signature for scorer"""
    ftr_type_to_scorer_tensor_spec = TASK_TYPE_TO_SCORER_TENSOR_SPEC[task_type]
    input_signature = {}
    if has_query:
        ftr_type = InternalFtrType.QUERY_FTRS
        input_signature[ftr_type] = ftr_type_to_scorer_tensor_spec[ftr_type](num_deep_ftrs)

    if num_doc_fields:
        ftr_type = InternalFtrType.DOC_FTRS
        input_signature[ftr_type] = ftr_type_to_scorer_tensor_spec[ftr_type](num_doc_fields, num_deep_ftrs)

    if num_user_fields:
        ftr_type = InternalFtrType.USER_FTRS
        input_signature[ftr_type] = ftr_type_to_scorer_tensor_spec[ftr_type](num_user_fields, num_deep_ftrs)

    ftr_type = InputFtrType.DENSE_FTRS_COLUMN_NAMES
    if ftr_type in feature_type_to_name:
        for ftr_name in feature_type_to_name[ftr_type]:
            input_signature[ftr_name] = ftr_type_to_scorer_tensor_spec[ftr_type](ftr_name, feature_name_to_num[ftr_name])

    ftr_type = InputFtrType.SPARSE_FTRS_COLUMN_NAMES
    if ftr_type in feature_type_to_name:
        for ftr_name in feature_type_to_name[ftr_type]:
            input_signature[ftr_name] = ftr_type_to_scorer_tensor_spec[ftr_type](feature_name_to_num[ftr_name])

    ftr_type = InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES
    if ftr_type in feature_type_to_name:
        for ftr_name in feature_type_to_name[ftr_type]:
            input_signature[ftr_name] = ftr_type_to_scorer_tensor_spec[ftr_type](feature_name_to_num[ftr_name])

    if InputFtrType.TASK_ID_COLUMN_NAME in feature_type_to_name:
        ftr_type = InputFtrType.TASK_ID_COLUMN_NAME
        ftr_name = feature_type_to_name[ftr_type]
        input_signature[ftr_name] = ftr_type_to_scorer_tensor_spec[ftr_type](ftr_name)

    return [input_signature]


class DetextModel(tf.keras.models.Model):
    """DeText model. Please check the DeText design doc for more details """

    def __init__(self, feature_type2name, feature_name2num: dict,
                 task_type,
                 use_deep,
                 use_dense_ftrs,
                 use_sparse_ftrs,
                 sparse_embedding_size,
                 sparse_embedding_cross_ftr_combiner,
                 sparse_embedding_same_ftr_combiner,
                 num_hidden,
                 num_classes,
                 task_ids,
                 has_query,
                 emb_sim_func,
                 ftr_mean, ftr_std,
                 rescale_dense_ftrs,
                 rep_layer_param
                 ):
        """ Initializes DetextModel

        :param feature_type2name: Dict mapping feature types to feature names
        :param task_type: Type of task in process: RANKING/CLASSIFICATION

        Check args.py for details of other params
        """
        super().__init__()

        self._task_type = task_type
        self._feature_type2name = feature_type2name
        self._feature_name2num = feature_name2num

        self._use_deep = use_deep
        self._use_dense_ftrs = use_dense_ftrs
        self._use_sparse_ftrs = use_sparse_ftrs
        self._use_wide = use_dense_ftrs or use_sparse_ftrs
        self._use_shallow_tower_sparse_ftrs = InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES in feature_type2name

        self._task_ids = task_ids if task_ids is not None else [0]
        self._num_classes = num_classes

        self._rescale_dense_ftrs = rescale_dense_ftrs
        self._nums_dense_ftrs = get_feature_nums(feature_name2num, feature_type2name, InputFtrType.DENSE_FTRS_COLUMN_NAMES)
        self._total_num_dense_ftrs = sum(self._nums_dense_ftrs)
        self._nums_sparse_ftrs = get_feature_nums(feature_name2num, feature_type2name, InputFtrType.SPARSE_FTRS_COLUMN_NAMES)
        self._num_hidden = num_hidden
        self._activations = ['tanh'] * len(num_hidden)
        self._nums_shallow_tower_sparse_ftrs = get_feature_nums(feature_name2num, feature_type2name, InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES)

        self.feature_name_type_converter = FeatureNameTypeConverter(task_type, feature_type2name)
        self.feature_grouper = FeatureGrouper()

        self._normalize_dense_ftrs = ftr_mean is not None
        if self._normalize_dense_ftrs:
            self.dense_ftrs_normalizer = FeatureNormalizer(ftr_mean=ftr_mean, ftr_std=ftr_std)

        if self._rescale_dense_ftrs and self._use_dense_ftrs:
            self.dense_ftrs_rescaler = FeatureRescaler(self._total_num_dense_ftrs, prefix='dense_ftrs_')

        if self._use_sparse_ftrs:
            self.sparse_embedding_layer = SparseEmbeddingLayer(sparse_embedding_size=sparse_embedding_size,
                                                               nums_sparse_ftrs=self._nums_sparse_ftrs,
                                                               sparse_embedding_cross_ftr_combiner=sparse_embedding_cross_ftr_combiner,
                                                               sparse_embedding_same_ftr_combiner=sparse_embedding_same_ftr_combiner,
                                                               initializer='glorot_uniform'
                                                               )

        if self._use_shallow_tower_sparse_ftrs:
            self.shallow_tower = ShallowTowerLayer(self._nums_shallow_tower_sparse_ftrs, num_classes)

        self.num_sim_ftrs = 0
        if use_deep is True:
            self.representation_layer = RepresentationLayer(**rep_layer_param)

        self.interaction_layer = InteractionLayer(use_deep_ftrs=use_deep,
                                                  use_wide_ftrs=self._use_wide,
                                                  task_ids=task_ids,
                                                  num_hidden=num_hidden,
                                                  activations=self._activations,

                                                  num_user_fields_for_interaction=self.representation_layer.output_num_user_fields if use_deep else 0,
                                                  num_doc_fields_for_interaction=self.representation_layer.output_num_doc_fields if use_deep else 0,
                                                  has_query=has_query,
                                                  emb_sim_func=emb_sim_func,
                                                  deep_ftrs_size=self.representation_layer.ftr_size if use_deep else 0)
        self.scoring_layer = ScoringLayer(task_ids, num_classes)
        self.output_transform_layer = OutputTransformLayer(task_type)

        self._scorer_input_signature = create_scorer_input_signature(
            task_type=self._task_type,
            feature_type_to_name=self._feature_type2name,
            feature_name_to_num=self._feature_name2num,
            num_doc_fields=self.representation_layer.output_num_doc_fields if use_deep else 0,
            num_user_fields=self.representation_layer.output_num_user_fields if use_deep else 0,
            num_deep_ftrs=self.representation_layer.ftr_size if use_deep else 0,
            has_query=InputFtrType.QUERY_COLUMN_NAME in self._feature_type2name,
        )
        self.generate_scores_given_deep_representations = tf.function(func=partial(self._generate_scores_given_deep_representations, training=False),
                                                                      input_signature=self._scorer_input_signature)

        if use_deep:
            self._embedding_generator_input_signature = create_embedding_generator_signature(
                feature_type2name=self._feature_type2name,
                task_type=self._task_type
            )
            self.generate_deep_representations = tf.function(func=partial(self._generate_deep_representations, training=False),
                                                             input_signature=self._embedding_generator_input_signature)

        self._training_output_name = TASK_TYPE_TO_TRAINING_OUTPUT_NAME[task_type]

    def call(self, inputs, training=False, **kwargs):
        """Feeds input through DeText model graph and returns the output

        Checks parsing_utils.InputFtrType for legit feature type inputs and FeatureArgs in args.py for acceptable inputs

        :return Map {output_name: output}
        """
        inputs = inputs.copy()

        if self._use_deep:
            deep_inputs = {}
            for ftr_type in DEEP_FTR_TYPES:
                ftr_name_list = as_list(self._feature_type2name.get(ftr_type, []))
                for ftr_name in ftr_name_list:
                    deep_inputs[ftr_name] = inputs.pop(ftr_name)
            inputs.update(self._generate_deep_representations(deep_inputs, training=training))

        return self._generate_scores_given_deep_representations(inputs)

    def _generate_deep_representations(self, inputs, training=False):
        """Generates representations for deep features such as text and id features"""
        deep_features_inputs = self.feature_name_type_converter({InternalFtrType.DEEP_FTR_BAG: inputs})[InternalFtrType.DEEP_FTR_BAG]
        return self.representation_layer(deep_features_inputs, training=training)

    def _generate_scores_given_deep_representations(self, inputs, training=False):
        """Returns scores computed given wide features and pre-computed deep embeddings"""
        multitask_features_inputs = self.feature_name_type_converter({InternalFtrType.MULTITASK_FTR_BAG: inputs})[InternalFtrType.MULTITASK_FTR_BAG]

        interaction_layer_inputs = {**inputs, **multitask_features_inputs}
        if self._use_wide:
            interaction_layer_inputs[InternalFtrType.WIDE_FTRS] = self.transform_wide_features(inputs)
        interaction_layer_outputs = self.interaction_layer(interaction_layer_inputs)

        scoring_layer_inputs = {**multitask_features_inputs, **interaction_layer_outputs}
        scores = self.scoring_layer(scoring_layer_inputs)
        if self._use_shallow_tower_sparse_ftrs:
            wide_features_inputs = self.feature_name_type_converter({InternalFtrType.WIDE_FTR_BAG: inputs})[InternalFtrType.WIDE_FTR_BAG]
            shallow_tower_inputs = {
                InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES: wide_features_inputs[InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES]}
            shallow_tower_offset = self.shallow_tower(shallow_tower_inputs)
            scores += shallow_tower_offset
        return self.output_transform_layer(scores)

    def transform_wide_features(self, inputs):
        """Transforms sparse and dense features and concatenates them into wide features"""
        wide_features_inputs = self.feature_name_type_converter({InternalFtrType.WIDE_FTR_BAG: inputs})[InternalFtrType.WIDE_FTR_BAG]
        wide_features_inputs = self.feature_grouper(wide_features_inputs)

        wide_ftrs_list = list()
        if self._use_dense_ftrs:
            dense_ftrs = wide_features_inputs[InputFtrType.DENSE_FTRS_COLUMN_NAMES]
            wide_ftrs_list.append(self.transform_dense_ftrs(dense_ftrs))
        if self._use_sparse_ftrs:
            wide_ftrs_list.append(self.sparse_embedding_layer(wide_features_inputs))
        return tf.concat(wide_ftrs_list, axis=-1)

    @tf.function
    def generate_training_scores(self, inputs, training=False, **kwargs):
        """Feeds input through DeText model graph and returns the scores for training （loss computation）

        Checks parsing_utils.InputFtrType for legit feature type inputs and FeatureArgs in args.py for acceptable inputs
        :return output
        """
        return self(inputs, training=training, **kwargs)[self._training_output_name]

    def transform_dense_ftrs(self, dense_ftrs):
        """Processes dense features with nan padding, normalization and rescaling"""
        dense_ftrs = self.pad_dense_ftrs_nan(dense_ftrs)
        if self._normalize_dense_ftrs:
            dense_ftrs = self.dense_ftrs_normalizer(dense_ftrs)

        if self._rescale_dense_ftrs:
            dense_ftrs = self.dense_ftrs_rescaler(dense_ftrs)
        return dense_ftrs

    @staticmethod
    def pad_dense_ftrs_nan(dense_ftrs):
        """ Pads NaN in dense features with zeros """
        return tf.compat.v1.where(tf.math.is_nan(dense_ftrs), tf.zeros_like(dense_ftrs), dense_ftrs)


def create_detext_model(feature_type2name, feature_name2num,
                        task_type,
                        use_deep, use_dense_ftrs,
                        use_sparse_ftrs,
                        sparse_embedding_size,
                        sparse_embedding_cross_ftr_combiner,
                        sparse_embedding_same_ftr_combiner,
                        num_hidden,
                        num_classes,
                        task_ids,
                        has_query,
                        emb_sim_func,
                        ftr_mean, ftr_std,
                        rescale_dense_ftrs,
                        rep_layer_param):
    """Creates a DetextModel instance for given params"""
    keras_inputs = create_keras_inputs(feature_type2name, feature_name2num, task_type)
    detext_model = DetextModel(feature_type2name=feature_type2name,
                               feature_name2num=feature_name2num,
                               task_type=task_type,
                               use_deep=use_deep, use_dense_ftrs=use_dense_ftrs,
                               use_sparse_ftrs=use_sparse_ftrs,
                               sparse_embedding_size=sparse_embedding_size,
                               sparse_embedding_cross_ftr_combiner=sparse_embedding_cross_ftr_combiner,
                               sparse_embedding_same_ftr_combiner=sparse_embedding_same_ftr_combiner,
                               num_hidden=num_hidden,
                               num_classes=num_classes,
                               task_ids=task_ids,
                               has_query=has_query,
                               emb_sim_func=emb_sim_func,
                               ftr_mean=ftr_mean, ftr_std=ftr_std,
                               rescale_dense_ftrs=rescale_dense_ftrs,
                               rep_layer_param=rep_layer_param)
    detext_model(keras_inputs)
    return detext_model
