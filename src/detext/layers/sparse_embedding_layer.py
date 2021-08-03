import tensorflow as tf

from detext.utils.parsing_utils import InputFtrType


class SparseEmbeddingLayer(tf.keras.layers.Layer):
    """Sparse embedding layer that accepts a sparse tensor as input, looks up embeddings and combines embeddings using given combiner"""

    def __init__(self, sparse_embedding_size, nums_sparse_ftrs, initializer, sparse_embedding_cross_ftr_combiner, sparse_embedding_same_ftr_combiner):
        """ Initializes SparseEmbeddingLayer

        :param sparse_embedding_size: size of sparse embedding
        :param nums_sparse_ftrs: numbers of sparse features for each column
        :param initializer: initializer for embeddings
        :param sparse_embedding_cross_ftr_combiner: how to combine the column embeddings. E.g., sum
        :param sparse_embedding_same_ftr_combiner: how to combine the look up embeddings within the column. E.g., sum/mean
        """
        super().__init__()
        self._sparse_embedding_cross_ftr_combiner = sparse_embedding_cross_ftr_combiner
        self._sparse_embedding_same_ftr_combiner = sparse_embedding_same_ftr_combiner
        self._nums_sparse_ftrs = nums_sparse_ftrs

        combiner2sparse_embedding_size = {
            'sum': sparse_embedding_size,
            'concat': sparse_embedding_size * len(nums_sparse_ftrs)
        }
        self._sparse_embedding_size = combiner2sparse_embedding_size[sparse_embedding_cross_ftr_combiner]
        self._combiner_fn = self._get_embedding_combiner_fn(sparse_embedding_cross_ftr_combiner)

        for i, num_sparse_ftrs in enumerate(nums_sparse_ftrs):
            setattr(self, self._get_embedding_weights_name(i), self.add_weight(
                name=self._get_embedding_weights_name(i),
                shape=[num_sparse_ftrs, sparse_embedding_size],
                dtype=tf.dtypes.float32,
                initializer=initializer, trainable=True
            ))

    def _get_embedding_combiner_fn(self, sparse_embedding_combiner):
        def concat_embedding(lst):
            return tf.concat(lst, axis=-1)

        def sum_embedding(lst):
            return tf.reduce_sum(lst, axis=0)

        combiner2embedding_combiner_fn = {
            'sum': sum_embedding,
            'concat': concat_embedding
        }

        return combiner2embedding_combiner_fn[sparse_embedding_combiner]

    def _get_embedding_weights_name(self, i):
        return f"sparse_embedding_weights_{i}"

    def _get_embedding_weights(self, i):
        return getattr(self, self._get_embedding_weights_name(i))

    def call(self, inputs, **kwargs):
        """Looks up and combines embeddings corresponding to given sparse features

        :param inputs: Map containing {
          InputFtrType.SPARSE_FTRS_COLUMN_NAMES: List(tf.SparseFeature)
        }. The last dimension of the sparse feature should be <= total_num_sparse_ftrs
        """
        sparse_ftrs = inputs[InputFtrType.SPARSE_FTRS_COLUMN_NAMES]
        sparse_embeddings = []
        for i, sparse_ftr in enumerate(sparse_ftrs):
            dense_shape = tf.shape(sparse_ftr, out_type=tf.dtypes.int64)
            values = sparse_ftr.indices[:, -1]
            sparse_ids = tf.sparse.SparseTensor(indices=sparse_ftr.indices, values=values,
                                                dense_shape=dense_shape)
            sparse_weights = sparse_ftr
            sparse_embedding = tf.nn.safe_embedding_lookup_sparse(self._get_embedding_weights(i), sparse_ids=sparse_ids,
                                                                  sparse_weights=sparse_weights, combiner=self._sparse_embedding_same_ftr_combiner)
            sparse_embeddings.append(sparse_embedding)

        return self._combiner_fn(sparse_embeddings)
