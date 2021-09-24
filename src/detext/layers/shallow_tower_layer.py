import tensorflow as tf

from .sparse_embedding_layer import SparseEmbeddingLayer
from detext.utils.parsing_utils import InputFtrType


class ShallowTowerLayer(tf.keras.layers.Layer):
    """Shallow tower layer including only linear combination of features """

    def __init__(self, nums_shallow_tower_sparse_ftrs, num_classes, initializer='glorot_uniform'):
        super(ShallowTowerLayer, self).__init__()
        self.sparse_linear = SparseEmbeddingLayer(num_classes, nums_shallow_tower_sparse_ftrs, initializer, 'sum', 'sum')

    def call(self, inputs, **kwargs):
        sparse_ftrs = inputs[InputFtrType.SHALLOW_TOWER_SPARSE_FTRS_COLUMN_NAMES]
        return self.sparse_linear({InputFtrType.SPARSE_FTRS_COLUMN_NAMES: sparse_ftrs})
