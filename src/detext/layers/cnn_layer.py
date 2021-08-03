"""
The model that uses CNN to extract features from texts, and generate a score for a query/doc pair
"""

import tensorflow as tf

from detext.layers.embedding_layer import create_embedding_layer
from detext.utils.layer_utils import get_sorted_dict
from detext.utils.parsing_utils import InputFtrType, InternalFtrType


class CnnLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filter_window_sizes,
                 num_filters,
                 num_user_fields,
                 num_doc_fields,
                 min_len,
                 max_len,
                 embedding_layer_param,
                 embedding_hub_url,
                 **kwargs):
        """ CNN text encoder

        For more details on parameters, check args.py
        """
        super(CnnLayer, self).__init__()
        self._filter_window_sizes = filter_window_sizes
        self._num_filters = num_filters
        self._num_doc_fields = num_doc_fields
        self._num_user_fields = num_user_fields
        self.text_ftr_size = len(filter_window_sizes) * num_filters

        self.min_len = tf.constant(min_len, dtype=tf.dtypes.int32)
        self.max_len = tf.constant(max_len, dtype=tf.dtypes.int32)
        self.num_cls_sep = tf.constant(max(1, max(filter_window_sizes) - 1), dtype=tf.dtypes.int32)

        self.embedding = create_embedding_layer(embedding_layer_param, embedding_hub_url)
        self._num_units = self.embedding.num_units().numpy()

        self.query_encoders = self._create_conv_layer("query")

        self.user_encoders = []
        for i in range(num_user_fields):
            self.user_encoders.append(self._create_conv_layer(f"user_{i}"))

        self.doc_encoders = []
        for i in range(num_doc_fields):
            self.doc_encoders.append(self._create_conv_layer(f"doc_{i}"))

    def _create_conv_layer(self, name_prefix):
        return create_conv_layer(name_prefix, self._filter_window_sizes, self._num_units, self._num_filters)

    def call(self, inputs, **kwargs):
        """Applies CNN to convert text to a fix length embedding

        :param inputs: Dict A mapping that contains the following key:
            query: Tensor(dtype=tf.string)  Shape=[batch_size]
            doc_fields: list(Tensor(dtype=string))/Tensor  A list of document fields. Each has shape=
                [batch_size, max_group_size]. For online scoring, these fields may be precomputed and
                input as Tensor
            user_fields: list(Tensor(dtype=string))/Tensor  A list of user fields. Each has shape=
                [batch_size]. For online scoring, these fields may be precomputed and
                input as Tensor
        :return:
        """
        query = inputs.get(InputFtrType.QUERY_COLUMN_NAME, None)
        doc_fields = inputs.get(InputFtrType.DOC_TEXT_COLUMN_NAMES, None)
        user_fields = inputs.get(InputFtrType.USER_TEXT_COLUMN_NAMES, None)

        query_ftrs = self._apply_cnn_on_query(query) if query is not None else None
        user_ftrs = self._apply_cnn_on_user(user_fields) if user_fields is not None else None
        doc_ftrs = self._apply_cnn_on_doc(doc_fields) if doc_fields is not None else None

        return query_ftrs, doc_ftrs, user_ftrs

    def _get_embedding_input(self, sentences):
        return get_sorted_dict(
            {InternalFtrType.SENTENCES: sentences,
             InternalFtrType.MIN_LEN: self.min_len,
             InternalFtrType.MAX_LEN: self.max_len,
             InternalFtrType.NUM_CLS: self.num_cls_sep,
             InternalFtrType.NUM_SEP: self.num_cls_sep}
        )

    def _apply_cnn_on_query(self, query):
        """
        Apply cnn on queries

        :return Tensor  Query features. Shape=[batch_size, num_filters*len(filter_window_size)]
        """
        embedding_result = self.embedding(self._get_embedding_input(query))
        query_emb = embedding_result[InternalFtrType.EMBEDDED]
        query_length = embedding_result[InternalFtrType.LENGTH]
        query_ftrs = self._apply_cnn_on_text(query_emb,
                                             self._filter_window_sizes,
                                             self.query_encoders,
                                             query_length)
        return query_ftrs

    def _apply_cnn_on_user(self, user_fields):
        """
        Apply cnn on user fields

        :return Tensor  User features. Shape=[batch_size, num_user_fields, num_filters*len(filter_window_size)]
        """
        if type(user_fields) is not list:
            return user_fields

        user_ftrs = []
        for i, user_field in enumerate(user_fields):
            embedding_result = self.embedding(self._get_embedding_input(user_field))
            user_emb = embedding_result[InternalFtrType.EMBEDDED]
            user_length = embedding_result[InternalFtrType.LENGTH]
            user_field_ftrs = self._apply_cnn_on_text(user_emb,
                                                      self._filter_window_sizes,
                                                      self.user_encoders[i],
                                                      user_length)
            user_ftrs.append(user_field_ftrs)
        user_ftrs = tf.stack(user_ftrs, axis=1)  # shape=[batch_size, num_user_fields, num_filters*len(filter_window_size)]
        return user_ftrs

    def _apply_cnn_on_doc(self, doc_fields):
        """Apply cnn on documents

        :return Tensor  Document features. Shape=[batch_size, max_group_size, num_doc_fields, num_ftrs]
        """
        if type(doc_fields) is not list:
            return doc_fields

        doc_ftrs = []
        for i, doc_field in enumerate(doc_fields):
            doc_field_shape = tf.shape(doc_field)
            doc_field_reshape = tf.reshape(doc_field, shape=[doc_field_shape[0] * doc_field_shape[1]])

            embedding_result = self.embedding(self._get_embedding_input(doc_field_reshape))
            doc_emb_reshape = embedding_result[InternalFtrType.EMBEDDED]
            doc_length = embedding_result[InternalFtrType.LENGTH]
            doc_field_ftrs = self._apply_cnn_on_text(doc_emb_reshape,
                                                     self._filter_window_sizes,
                                                     self.doc_encoders[i],
                                                     doc_length)
            # restore batch_size and group_size
            doc_field_ftrs = tf.reshape(doc_field_ftrs, shape=[doc_field_shape[0], doc_field_shape[1], self.text_ftr_size])
            doc_ftrs.append(doc_field_ftrs)
        doc_ftrs = tf.stack(doc_ftrs, axis=2)  # shape=[batch_size, max_group_size, num_doc_fields, num_ftrs]
        return doc_ftrs

    def _apply_cnn_on_text(self,
                           text_emb,
                           filter_window_sizes,
                           conv_layers,
                           text_length=None,
                           set_empty_zero=False
                           ):
        """For any text, apply several cnn with the same window size """
        return apply_cnn_on_text(text_emb, filter_window_sizes, conv_layers, text_length, set_empty_zero)


def apply_cnn_on_text(text_emb,
                      filter_window_sizes,
                      conv_layers,
                      text_length,
                      set_empty_zero):
    """
    For any text, apply several cnn with the same window size.
    The final embedding is concatenated from the cnn outputs.

    :return Tensor  Text features. Shape=[batch_size, num_filters*len(filter_window_sizes)]
    """
    text_ftrs = []
    for filter_size, conv_layer in zip(filter_window_sizes, conv_layers):
        t_conv = conv_layer(tf.expand_dims(text_emb, axis=3))
        if text_length is not None:
            # Make sure the convolution results is consistent
            mask_length = tf.maximum(text_length - filter_size + 1, 0)
            mask = tf.sequence_mask(mask_length, maxlen=tf.shape(input=text_emb)[1] - filter_size + 1, dtype=tf.float32)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=2), axis=2)
            t_conv *= mask

        t_pool = tf.reduce_max(input_tensor=t_conv, axis=1)  # shape = [batch_size, 1, num_filters]
        t_pool = tf.squeeze(t_pool, axis=1)  # shape = [batch_size, num_filters]

        # set the cnn output of empty strings as 0
        if set_empty_zero:
            not_empty_str = tf.cast(tf.not_equal(text_length, max(filter_window_sizes) * 2 - 2), dtype=tf.float32)
            t_pool *= tf.expand_dims(not_empty_str, axis=1)

        text_ftrs.append(t_pool)
    text_ftrs = tf.concat(text_ftrs, 1)  # shape = [batch_size, num_filters*len(filter_window_sizes)]
    return text_ftrs


def create_conv_layer(name_prefix, filter_window_sizes, num_units, num_filters):
    """Returns a list of cnn layers with given window sizes """
    return [
        tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[filter_size, num_units],
            activation=tf.nn.relu,
            padding='valid',
            name=f'{name_prefix}_{i}')
        for i, filter_size in enumerate(filter_window_sizes)
    ]
