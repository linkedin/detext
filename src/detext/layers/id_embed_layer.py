import tensorflow as tf

from detext.layers.embedding_layer import create_embedding_layer
from detext.utils.parsing_utils import InputFtrType, InternalFtrType

DEFAULT_MIN_LEN = 1
DEFAULT_MAX_LEN = 100


class IdEmbedLayer(tf.keras.layers.Layer):
    """ ID embedding layer"""

    def __init__(self, num_id_fields, embedding_layer_param, embedding_hub_url_for_id_ftr):
        """ Initializes the layer

        For more details on parameters, check args.py
        """
        super(IdEmbedLayer, self).__init__()
        self._num_id_fields = num_id_fields

        self.min_len = DEFAULT_MIN_LEN
        self.max_len = DEFAULT_MAX_LEN
        self.num_cls_sep = 0

        if num_id_fields:
            self.embedding = create_embedding_layer(embedding_layer_param, embedding_hub_url_for_id_ftr)
            self.id_ftr_size = self.embedding.num_units()

    def call(self, inputs, **kwargs):
        """ Applies ID embedding lookup and summation on document and user fields

        :param inputs: Dict A mapping that contains the following key:
            doc_id_fields: list(Tensor(dtype=string))  List of document fields. Each has shape=[batch_size, max_group_size]
            user_id_fields: list(Tensor(dtype=string))  List of user fields. Each has shape=[batch_size]
        :return: doc_ftrs, user_ftrs
        """
        doc_id_fields = inputs.get(InputFtrType.DOC_ID_COLUMN_NAMES, None)
        user_id_fields = inputs.get(InputFtrType.USER_ID_COLUMN_NAMES, None)

        if self._num_id_fields == 0:
            assert doc_id_fields is None and user_id_fields is None, "Document ID fields and user ID fields must be None when there's no id field"

        user_ftrs = self.apply_embed_on_user_id(user_id_fields) if user_id_fields is not None else None
        doc_ftrs = self.apply_embed_on_doc_id(doc_id_fields) if doc_id_fields is not None else None
        return doc_ftrs, user_ftrs

    def apply_embedding(self, inputs):
        """Applies embedding on give inputs

        :param inputs Tensor(dtype=string) Shape=[batch_size]
        :return Tensor(dtype=string) Shape=[batch_size, sentence_len, num_units_for_id_ftr]
        """
        embedding_result = self.embedding({
            InternalFtrType.SENTENCES: inputs,
            InternalFtrType.NUM_CLS: self.num_cls_sep,
            InternalFtrType.NUM_SEP: self.num_cls_sep,
            InternalFtrType.MIN_LEN: self.min_len,
            InternalFtrType.MAX_LEN: self.max_len,
        })

        seq_length = embedding_result[InternalFtrType.LENGTH]
        max_seq_len = tf.math.reduce_max(seq_length)
        seq_mask = tf.expand_dims(tf.sequence_mask(seq_length, max_seq_len, dtype=tf.float32), axis=-1)
        seq_length = tf.expand_dims(tf.cast(seq_length, dtype=tf.dtypes.float32), axis=-1)

        user_id_embeddings = embedding_result[InternalFtrType.EMBEDDED]
        sum_user_id_embedding = tf.reduce_sum(
            input_tensor=user_id_embeddings * seq_mask, axis=1)  # [batch_size, num_units_for_id_ftr]
        user_id_avg_embedding = tf.math.divide_no_nan(sum_user_id_embedding, seq_length)  # [batch_size, num_units_for_id_ftr]
        return user_id_avg_embedding

    def apply_embed_on_user_id(self, user_id_fields):
        """Applies embedding lookup and averaging for user id features

        :return Tensor Shape=[batch_size, num_user_id_fields, num_units_for_id_ftr]
        """
        user_ftrs = []
        for i, user_field in enumerate(user_id_fields):
            user_id_avg_embedding = self.apply_embedding(user_field)
            user_ftrs.append(user_id_avg_embedding)
        return tf.stack(user_ftrs, axis=1)

    def apply_embed_on_doc_id(self, doc_id_fields):
        """Applies embedding lookup and averaging for doc id features

        :return Tensor Shape=[batch_size, max_group_size, num_doc_id_fields, num_units_for_id_ftr]
        """
        doc_ftrs = []
        for i, doc_field in enumerate(doc_id_fields):
            doc_field_shape = tf.shape(doc_field)
            reshape_doc_field = tf.reshape(doc_field, shape=[doc_field_shape[0] * doc_field_shape[1]])
            doc_id_avg_embedding = self.apply_embedding(reshape_doc_field)
            doc_id_avg_embedding = tf.reshape(doc_id_avg_embedding, shape=[doc_field_shape[0], doc_field_shape[1], self.id_ftr_size])
            doc_ftrs.append(doc_id_avg_embedding)
        return tf.stack(doc_ftrs, axis=2)
