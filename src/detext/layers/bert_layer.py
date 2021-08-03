import tensorflow as tf
import tensorflow_hub as hub

from libert.preprocess import BertPreprocessLayer

from detext.utils.parsing_utils import InputFtrType

BERT_VAR_PREFIX = "bert"
SENTENCEPIECE = "sentencepiece"
SPACE = "space"
WORDPIECE = "wordpiece"


class BertLayer(tf.keras.layers.Layer):
    """BERT text encoder"""

    def __init__(self, num_units, CLS, SEP, PAD, UNK, min_len, max_len, bert_hub_url, **kwargs):
        """ Initializes the layer

        For more details on parameters, check the argument parser in run_detext.py
        """
        super(BertLayer, self).__init__()

        self.text_ftr_size = num_units
        # Load pretrained model using hub
        with tf.name_scope(BERT_VAR_PREFIX):
            self.text_encoder = hub.KerasLayer(hub.resolve(bert_hub_url), trainable=True)

        self.last_layer_cls = tf.keras.layers.Dense(num_units, activation='tanh', use_bias=True, name=BERT_VAR_PREFIX + '_last_layer_cls')
        self.preprocess_layer = BertPreprocessLayer(self.text_encoder, max_len, min_len, CLS, SEP, PAD, UNK)
        self._pad_id = self.preprocess_layer.pad_id()
        self._pad_id = tf.cast(self._pad_id, tf.int32)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def _preprocess_query(self, query):
        """
        Preprocess queries

        :return Tensor(dtype=tf.int)  Shape=[batch_size, query_length]
        """
        # Copy input tensor to prevent preprocess layer from changing the original tensor
        query = tf.identity(query)
        return self.preprocess_layer(query)

    def _preprocess_user(self, user_fields):
        """
        Preprocess user fields

        :return list(Tensor(dtype=int))/Tensor  A list of user fields. Each has shape=
                [batch_size, user_field_length].
        """
        if type(user_fields) is not list:
            return user_fields

        # Copy input tensor to prevent preprocess layer from changing the original tensor
        user_fields = tf.nest.map_structure(tf.identity, user_fields)

        for i, user_field in enumerate(user_fields):
            user_fields[i] = self.preprocess_layer(user_field)

        return user_fields

    def _preprocess_doc(self, doc_fields):
        """
        Preprocess doc fields

        :return list(Tensor(dtype=int))/Tensor  A list of document fields. Each has shape=
                [batch_size, max_group_size, doc_field_length].
        """
        if type(doc_fields) is not list:
            return doc_fields

        # Copy input tensor to prevent preprocess layer from changing the original tensor
        doc_fields = tf.nest.map_structure(tf.identity, doc_fields)

        for i, doc_field in enumerate(doc_fields):
            doc_field_shape = tf.shape(doc_field)
            doc_field_reshape = tf.reshape(doc_field, shape=[doc_field_shape[0] * doc_field_shape[1]])
            doc_field_preprocessed = self.preprocess_layer(doc_field_reshape)
            doc_field = tf.reshape(doc_field_preprocessed, shape=[doc_field_shape[0], doc_field_shape[1], -1])
            doc_fields[i] = doc_field

        return doc_fields

    def call(self, inputs, training=None, **kwargs):
        """ Apply BERT to convert text to features

        :param inputs: Dict A mapping that contains the following key:
            query: Tensor(dtype=tf.string)  Shape=[batch_size]
            doc_fields: list(Tensor(dtype=string))/Tensor  A list of document fields. Each has shape=
                [batch_size, max_group_size]. For online scoring, these fields may be precomputed and
                input as Tensor
            user_fields: list(Tensor(dtype=string))/Tensor  A list of user fields. Each has shape=
                [batch_size]. For online scoring, these fields may be precomputed and
                input as Tensor
        :param training: bool Whether current mode is training

        :return query_ftrs,doc_ftrs,user_ftrs  shape(query_ftrs)=[batch_size, num_units],
            shape(doc_ftrs)=[batch_size, max_group_size, num_doc_fields, num_units],
            shape(user_ftrs)=[batch_size, num_doc_fields, num_units]
        """

        query = inputs.get(InputFtrType.QUERY_COLUMN_NAME, None)
        doc_fields = inputs.get(InputFtrType.DOC_TEXT_COLUMN_NAMES, None)
        user_fields = inputs.get(InputFtrType.USER_TEXT_COLUMN_NAMES, None)

        query = self._preprocess_query(query) if query is not None else None
        user_fields = self._preprocess_user(user_fields) if user_fields is not None else None
        doc_fields = self._preprocess_doc(doc_fields) if doc_fields is not None else None

        has_query = query is not None
        has_doc = doc_fields is not None

        # at least one of query and doc_fields should be true
        assert has_query or has_doc, f'{InputFtrType.QUERY_COLUMN_NAME} and {InputFtrType.DOC_TEXT_COLUMN_NAMES} cannot both be None'

        # if doc is None, we cannot infer batch_size and max_group_size from doc
        batch_size = tf.shape(input=doc_fields[0])[0] if has_doc else tf.shape(input=query)[0]
        max_group_size = tf.shape(input=doc_fields[0])[1] if has_doc else None

        if not has_query:
            # Instead of removing the query sequence in the input id array fed into BERT, we use a fake query with
            #   the same size as the first document field in the first document. The content of this query will NOT
            #   be used anywhere. It is just to save effort on code changes handling the case when `query is None`
            query = tf.ones(dtype=tf.int32, shape=tf.shape(input=doc_fields[0][:, 0, :]))

        # 1. Combine query, user_fields and doc_fields into one matrix
        # Find largest text length value
        max_text_len, max_text_len_array = self.get_input_max_len(query, doc_fields, user_fields)

        bert_input_ids = self.get_bert_input_ids(query, doc_fields, user_fields, self._pad_id, max_text_len,
                                                 max_text_len_array)
        bert_input_mask = tf.cast(tf.not_equal(bert_input_ids, self._pad_id), dtype=tf.int32)

        # 2. Apply bert
        cls_output, seq_output = self.text_encoder([bert_input_ids, bert_input_mask, tf.zeros_like(bert_input_ids)])

        # 3. get query embedding and doc_fields embedding
        pooled_output = self.last_layer_cls(seq_output[:, 0, :])  # shape = [# of texts, bert_dim]

        pooled_output = self.dropout(pooled_output, training=training)

        # Get query features
        query_ftrs = pooled_output[0:batch_size] if has_query else None

        # Get document features
        if type(doc_fields) is not list:
            # If doc_fields is not a list, then it is a precomputed embedding tensor
            doc_ftrs = self.doc_fields
            doc_field_end = batch_size
        else:
            doc_field_start = batch_size
            doc_ftrs = []
            for _ in range(len(doc_fields)):
                doc_field_end = doc_field_start + batch_size * max_group_size
                dftrs = pooled_output[doc_field_start: doc_field_end]
                dftrs = tf.reshape(dftrs, shape=[batch_size, max_group_size, self.text_ftr_size])
                doc_ftrs.append(dftrs)
                doc_field_start = doc_field_end
            doc_ftrs = tf.stack(doc_ftrs, axis=2)  # shape = [batch_size, max_group_size, num_doc_fields, num_units]

        # Get user features
        if type(user_fields) is not list:
            user_ftrs = user_fields
        else:
            user_field_start = doc_field_end
            user_ftrs = []
            for _ in range(len(user_fields)):
                user_field_end = user_field_start + batch_size
                uftrs = pooled_output[user_field_start: user_field_end]
                user_ftrs.append(uftrs)
                user_field_start = user_field_end
            user_ftrs = tf.stack(user_ftrs, axis=1)  # shape=[batch_size, num_user_fields, num_units]

        return query_ftrs, doc_ftrs, user_ftrs

    @staticmethod
    def get_input_max_len(query, doc_fields, user_fields):
        max_text_len = tf.shape(input=query)[-1]
        max_text_len_array = [max_text_len]

        # Add doc field length info
        if type(doc_fields) is list:
            for doc_field in doc_fields:
                doc_field_text_len = tf.shape(input=doc_field)[-1]
                max_text_len_array.append(doc_field_text_len)
                max_text_len = tf.maximum(max_text_len, doc_field_text_len)

        # Add user field length info
        if type(user_fields) is list:
            for user_field in user_fields:
                user_field_text_len = tf.shape(input=user_field)[-1]
                max_text_len_array.append(user_field_text_len)
                max_text_len = tf.maximum(max_text_len, user_field_text_len)

        return max_text_len, max_text_len_array

    @staticmethod
    def get_bert_input_ids(query, doc_fields, user_fields, pad_id, max_text_len, max_text_len_array):
        padded_query = tf.pad(tensor=query, paddings=[[0, 0], [0, max_text_len - max_text_len_array[0]]],
                              constant_values=pad_id)
        bert_input_ids_array = [padded_query]

        if type(doc_fields) is list:
            for doc_field, doc_field_max_text_len in zip(doc_fields, max_text_len_array[1:]):
                # Shape=[batch_size*max_group_size, doc_field_max_text_len]
                doc_field_2d = tf.reshape(doc_field, shape=[-1, doc_field_max_text_len])
                padded_doc_field = tf.pad(
                    tensor=doc_field_2d, paddings=[[0, 0], [0, max_text_len - doc_field_max_text_len]],
                    constant_values=pad_id)
                bert_input_ids_array.append(padded_doc_field)

        if type(user_fields) is list:
            doc_field_end = 1 + len(doc_fields) if type(doc_fields) is list else 1

            for user_field, user_field_max_text_len in zip(user_fields, max_text_len_array[doc_field_end:]):
                # Shape=[batch_size, user_field_max_text_len]
                padded_user_field = tf.pad(tensor=user_field,
                                           paddings=[[0, 0], [0, max_text_len - user_field_max_text_len]],
                                           constant_values=pad_id)
                bert_input_ids_array.append(padded_user_field)

        bert_input_ids = tf.concat(bert_input_ids_array, axis=0)
        return bert_input_ids
