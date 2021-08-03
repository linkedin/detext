"""
Representation-based layers.
"""

import tensorflow as tf

from detext.layers import bert_layer
from detext.layers import cnn_layer
from detext.layers import id_embed_layer
from detext.layers import lstm_layer
from detext.utils.parsing_utils import InputFtrType, InternalFtrType

_FTR_EXT_NAME_TO_ENCODER = {
    'cnn': cnn_layer.CnnLayer,
    'lstm': lstm_layer.LstmLayer,
    'bert': bert_layer.BertLayer
}


class RepresentationLayer(tf.keras.layers.Layer):
    """
    The representation-based layer to generate deep features, based on the textual and id inputs of query, user fields, document fields.
    """

    def __init__(self, ftr_ext,
                 num_doc_fields,
                 num_user_fields,
                 num_doc_id_fields,
                 num_user_id_fields,
                 add_doc_projection,
                 add_user_projection,
                 text_encoder_param,
                 id_encoder_param):
        """ Initializes the layer

        :param text_encoder_param Mapping that contains parameters of the text encoder
        :param id_encoder_param Mapping that contains parameters of the id encoder
        For more details on parameters, check the argument parser in run_detext.py
        """
        super(RepresentationLayer, self).__init__()
        self._add_doc_projection = add_doc_projection
        self._add_user_projection = add_user_projection
        self._num_doc_fields = num_doc_fields
        self._num_user_fields = num_user_fields

        self.text_encoding_layer = _FTR_EXT_NAME_TO_ENCODER[ftr_ext](**text_encoder_param)
        self.id_encoding_layer = id_embed_layer.IdEmbedLayer(**id_encoder_param)
        self.ftr_size = self.text_encoding_layer.text_ftr_size

        if num_doc_id_fields + num_user_id_fields > 0:
            self.id2text_projection = tf.keras.layers.Dense(self.ftr_size)

        if add_doc_projection:
            self.doc_projection = tf.keras.layers.Dense(self.ftr_size, activation=tf.tanh, name="doc_ftrs_projection_layer")

        if add_user_projection:
            self.user_projection = tf.keras.layers.Dense(self.ftr_size, activation=tf.tanh, name="user_ftrs_projection_layer")

        self.tot_num_doc_fields = num_doc_fields + num_doc_id_fields
        self.tot_num_user_fields = num_user_fields + num_user_id_fields

        self.output_num_user_fields = 1 if add_user_projection else self.tot_num_user_fields
        self.output_num_doc_fields = 1 if add_doc_projection else self.tot_num_doc_fields

    def call(self, inputs, training=None, **kwargs):
        """ Merges text and id feature representation and computes similarity between user and document features

        :param inputs: Dict  A mapping that contains:
            query: Tensor(dtype=string)  Shape=[batch_size]
            doc_fields: list(Tensor(dtype=string))/Tensor  A list of document fields. Each has shape=
                [batch_size, max_group_size]. For online scoring, these fields may be precomputed and
                input as Tensor
            user_fields: list(Tensor(dtype=string))/Tensor  A list of user fields. Each has shape=
                [batch_size]. For online scoring, these fields may be precomputed and input as Tensor
            doc_id_fields: list(Tensor(dtype=string))  List of document fields. Each has
                shape=[batch_size, max_group_size]
            user_id_fields: list(Tensor(dtype=string))  List of user fields. Each has shape=[batch_size]
        :param training: boolean Whether it's under training mode
        :return: deep_ftrs dictionary containing query_ftrs, doc_ftrs and user_ftrs
        """
        query = inputs.get(InputFtrType.QUERY_COLUMN_NAME, None)
        doc_fields = inputs.get(InputFtrType.DOC_TEXT_COLUMN_NAMES, None)
        doc_id_fields = inputs.get(InputFtrType.DOC_ID_COLUMN_NAMES, None)
        user_fields = inputs.get(InputFtrType.USER_TEXT_COLUMN_NAMES, None)
        user_id_fields = inputs.get(InputFtrType.USER_ID_COLUMN_NAMES, None)

        query_ftrs, doc_ftrs, user_ftrs = self.text_encoding_layer(
            {InputFtrType.QUERY_COLUMN_NAME: query, InputFtrType.DOC_TEXT_COLUMN_NAMES: doc_fields, InputFtrType.USER_TEXT_COLUMN_NAMES: user_fields},
            training=training)
        doc_id_ftrs, user_id_ftrs = self.id_encoding_layer({InputFtrType.DOC_ID_COLUMN_NAMES: doc_id_fields, InputFtrType.USER_ID_COLUMN_NAMES: user_id_fields})

        query_ftrs, doc_ftrs, user_ftrs = self.get_composite_tensors(query_ftrs, doc_ftrs, user_ftrs, doc_id_ftrs, user_id_ftrs)
        query_ftrs, doc_ftrs, user_ftrs = self.add_projection(query_ftrs=query_ftrs, doc_ftrs=doc_ftrs, user_ftrs=user_ftrs)

        results = {InternalFtrType.QUERY_FTRS: query_ftrs,
                   InternalFtrType.DOC_FTRS: doc_ftrs,
                   InternalFtrType.USER_FTRS: user_ftrs}

        return {k: v for k, v in results.items() if v is not None}

    def get_composite_tensors(self, query_ftrs, doc_ftrs, user_ftrs, doc_id_ftrs, user_id_ftrs):
        """ Merges text and id features """
        field_names = ['query_ftrs', 'doc_ftrs', 'user_ftrs']
        text_ftrs = [query_ftrs, doc_ftrs, user_ftrs]
        id_ftrs = [None, doc_id_ftrs, user_id_ftrs]

        composite_tensors = []
        for field_name, text_field_tensor, id_field_tensor in zip(field_names, text_ftrs, id_ftrs):
            if id_field_tensor is not None:
                id_field_tensor = self.id2text_projection(id_field_tensor)

            # Compose a new tensor to include user id info and user text info
            # I.e., if there are 2 user id fields, 3 user text fields, we have 2 + 3 = 5 user embedding fields
            # Same processing also works for doc fields
            if text_field_tensor is not None and id_field_tensor is not None:
                composite_field_tensor = tf.concat([text_field_tensor, id_field_tensor], axis=-2)
            elif text_field_tensor is not None:
                composite_field_tensor = text_field_tensor
            else:
                composite_field_tensor = id_field_tensor

            # Assign a name to each text feature so that we can quickly locate them in the graph
            if composite_field_tensor is not None:
                composite_field_tensor = tf.identity(composite_field_tensor, name=field_name)

            composite_tensors.append(composite_field_tensor)

        query_ftrs, doc_ftrs, user_ftrs = composite_tensors
        return query_ftrs, doc_ftrs, user_ftrs

    def add_projection(self, query_ftrs, doc_ftrs, user_ftrs):
        """Projects user/document representations to one vector"""
        num_doc_fields = self.tot_num_doc_fields
        ftr_size = self.ftr_size
        batch_size, max_group_size = self.get_data_size(doc_ftrs)

        if self._add_doc_projection:
            doc_ftrs = tf.reshape(doc_ftrs, shape=[batch_size, max_group_size, 1, ftr_size * num_doc_fields])
            doc_ftrs = self.doc_projection(doc_ftrs)  # [batch_size, max_group_size, 1, ftr_size]
            doc_ftrs = tf.identity(doc_ftrs, name='doc_ftrs_projection')

        if self._add_user_projection:
            user_ftrs = tf.reshape(user_ftrs, shape=[batch_size, 1, ftr_size * self.tot_num_user_fields])
            user_ftrs = self.user_projection(user_ftrs)  # [batch_size, 1, ftr_size]
            user_ftrs = tf.identity(user_ftrs, name='user_ftrs_projection')

        return query_ftrs, doc_ftrs, user_ftrs

    def get_data_size(self, doc_ftrs):
        """Infers batch size, max group size"""
        assert doc_ftrs is not None, "doc_ftrs could not be None"
        data_shape = tf.shape(input=doc_ftrs)
        return data_shape[0], data_shape[1]
