import tensorflow as tf

from detext.layers.embedding_layer import create_embedding_layer
from detext.utils.layer_utils import get_sorted_dict
from detext.utils.parsing_utils import InputFtrType, InternalFtrType


class LstmLayer(tf.keras.layers.Layer):
    def __init__(self, bidirectional, rnn_dropout, num_layers, forget_bias,
                 min_len, max_len, embedding_layer_param, embedding_hub_url,
                 **kwargs):
        """ Initializes the model

        For more details on parameters, check the argument parser in run_detext.py
        """
        super(LstmLayer, self).__init__()
        self.min_len = tf.constant(min_len, dtype=tf.dtypes.int32)
        self.max_len = tf.constant(max_len, dtype=tf.dtypes.int32)
        self.num_cls_sep = tf.constant(1, dtype=tf.dtypes.int32)

        self._bidirectional = bidirectional
        self._forget_bias = forget_bias
        self._rnn_dropout = rnn_dropout

        self.forward_only = not bidirectional
        self.embedding = create_embedding_layer(embedding_layer_param, embedding_hub_url)
        self._num_units = self.embedding.num_units().numpy()
        self.text_ftr_size = self._num_units

        self.text_encoders = []
        for _ in range(num_layers):
            if self.forward_only:
                encoder = self.create_encoder(self._num_units)
            else:
                assert self._num_units % 2 == 0, "num_units must be a multiplier of 2 when bidirectional is True"
                fw_encoder = self.create_encoder(self._num_units // 2)
                bw_encoder = self.create_encoder(self._num_units // 2, go_backwards=True)
                encoder = tf.keras.layers.Bidirectional(fw_encoder, backward_layer=bw_encoder)

            self.text_encoders.append(encoder)

    def create_encoder(self, num_units, **kwargs):
        return tf.keras.layers.LSTM(num_units, bias_initializer=tf.keras.initializers.Constant(self._forget_bias), dropout=self._rnn_dropout,
                                    return_sequences=True, return_state=True, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        """ Apply LSTM on text fields

        :param query: Tensor(dtype=string)  Shape=[batch_size]
        :param doc_fields: list(Tensor(dtype=string))/Tensor  A list of document fields. Each has shape=
            [batch_size, max_group_size]. For online scoring, these fields may be precomputed and
            input as Tensor
        :param user_fields: list(Tensor(dtype=string))/Tensor  A list of user fields. Each has shape=
            [batch_size]. For online scoring, these fields may be precomputed and input as Tensor
        :param training: boolean Whether it's under training mode
        :return: query, document and user features
        """
        query = inputs.get(InputFtrType.QUERY_COLUMN_NAME, None)
        doc_fields = inputs.get(InputFtrType.DOC_TEXT_COLUMN_NAMES, None)
        user_fields = inputs.get(InputFtrType.USER_TEXT_COLUMN_NAMES, None)

        query_ftrs = self.apply_lstm_on_query(query, training) if query is not None else None
        user_ftrs = self.apply_lstm_on_user(user_fields, training) if user_fields is not None else None
        doc_ftrs = self.apply_lstm_on_doc(doc_fields, training) if doc_fields is not None else None
        return query_ftrs, doc_ftrs, user_ftrs

    def apply_lstm_on_query(self, query, training):
        """
        Applies LSTM on query

        :return Tensor  Query features. Shape=[batch_size, text_ftr_size]
        """
        return self.apply_lstm_on_text(query, training)[InternalFtrType.LAST_MEMORY_STATE]

    def apply_lstm_on_user(self, user_fields, training):
        """
        Applies LSTM on user fields

        :return Tensor  User features. Shape=[batch_size, num_user_fields, text_ftr_size]
        """
        if type(user_fields) is not list:
            return user_fields

        user_ftrs = []
        for i, user_field in enumerate(user_fields):
            user_field_ftrs = self.apply_lstm_on_text(user_field, training)[InternalFtrType.LAST_MEMORY_STATE]
            user_ftrs.append(user_field_ftrs)
        user_ftrs = tf.stack(user_ftrs, axis=1)  # shape=[batch_size, num_user_fields, text_ftr_size]
        return user_ftrs

    def apply_lstm_on_text(self, text, training):
        """ Applies LSTM on text with params partially filled with class members """
        return apply_lstm_on_text(text, self.text_encoders, self.embedding, self._bidirectional, self.min_len, self.max_len, self.num_cls_sep, training)

    def apply_lstm_on_doc(self, doc_fields, training):
        """ Applies LSTM on documents

        :return Tensor  Document features. Shape=[batch_size, max_group_size, num_doc_fields, text_ftr_size]
        """
        # doc_fields should be the doc embeddings
        if type(doc_fields) is not list:
            return doc_fields

        doc_ftrs = []
        for i, doc_field in enumerate(doc_fields):
            doc_shape = tf.shape(input=doc_field)  # Shape=[batch_size, group_size, sent_length]
            doc_field = tf.reshape(doc_field, shape=[doc_shape[0] * doc_shape[1]])
            doc_field_ftrs = self.apply_lstm_on_text(doc_field, training)[InternalFtrType.LAST_MEMORY_STATE]

            # Restore batch_size and group_size
            doc_field_ftrs = tf.reshape(doc_field_ftrs, shape=[doc_shape[0], doc_shape[1], self.text_ftr_size])
            doc_ftrs.append(doc_field_ftrs)
        doc_ftrs = tf.stack(doc_ftrs, axis=2)  # Shape=[batch_size, max_group_size, num_doc_fields, text_ftr_size]
        return doc_ftrs


def apply_lstm_on_embedding(input_emb, mask, lstm_encoders, forward_only, training):
    """Applies LSTM on given embeddings

    :param input_emb Tensor(dtype=float) Shape=[batch_size, sentence_len, num_units]
    :return A dictionary containing
        seq_outputs: Tensor(dtype=float) Shape=[batch_size, sentence_len, num_units]
        last_memory_state: Tensor(dtype=float) Shape=[batch_size, num_units]
        last_carry_state: Tensor(dtype=float) Shape=[batch_size, num_units]
    """
    first_encoder = lstm_encoders[0]
    mask = tf.cast(mask, dtype=tf.dtypes.bool)
    if forward_only:
        # Shape(seq_outputs) = [batch_size, seq_len, num_units]
        seq_outputs, last_memory_state, last_carry_state = first_encoder(input_emb, mask=mask, training=training)
        for encoder in lstm_encoders[1:]:  # Multi layer LSTM
            seq_outputs, last_memory_state, last_carry_state = encoder(seq_outputs, mask=mask, training=training)
    else:
        # Shape(seq_outputs) = [batch_size, seq_len, num_units]
        seq_outputs, last_fw_memory_state, last_fw_carry_state, last_bw_memory_state, last_bw_carry_state = first_encoder(input_emb, mask=mask,
                                                                                                                          training=training)
        for encoder in lstm_encoders[1:]:  # Multi layer LSTM
            seq_outputs, last_fw_memory_state, last_fw_carry_state, last_bw_memory_state, last_bw_carry_state = encoder(seq_outputs, mask=mask,
                                                                                                                        training=training)

        last_memory_state = tf.concat([last_fw_memory_state, last_bw_memory_state], axis=-1)
        last_carry_state = tf.concat([last_fw_carry_state, last_bw_carry_state], axis=-1)
    return {InternalFtrType.SEQ_OUTPUTS: seq_outputs,
            InternalFtrType.LAST_MEMORY_STATE: last_memory_state,
            InternalFtrType.LAST_CARRY_STATE: last_carry_state}


def apply_lstm_on_text(text, lstm_encoders, embedding_layer, bidirectional, min_len, max_len, num_cls_sep, training):
    """
    Applies LSTM on text

    :param text: Tensor Shape=[batch_size]
    :param lstm_encoders: List(LSTM) List of LSTMs that stacks sequentially
    :param embedding_layer Tensor Embedding matrix
    :param bidirectional boolean Indicator of whether it's bidirectional encoder
    :param training boolean Indicator of whether it's under training mode

    :return A dictionary containing
        seq_outputs: Tensor(dtype=float) Shape=[batch_size, sentence_len, num_units]
        last_memory_state: Tensor(dtype=float) Shape=[batch_size, num_units]
        last_carry_state: Tensor(dtype=float) Shape=[batch_size, num_units]
    """
    forward_only = not bidirectional

    input_seq = text
    inputs = get_sorted_dict(
        {InternalFtrType.SENTENCES: input_seq,
         InternalFtrType.NUM_CLS: num_cls_sep,
         InternalFtrType.NUM_SEP: num_cls_sep,
         InternalFtrType.MIN_LEN: min_len,
         InternalFtrType.MAX_LEN: max_len,
         }
    )

    embedding_result = embedding_layer(inputs)
    input_emb = embedding_result[InternalFtrType.EMBEDDED]
    seq_len = embedding_result[InternalFtrType.LENGTH]
    max_seq_len = tf.math.reduce_max(seq_len)
    mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)

    results = apply_lstm_on_embedding(input_emb, mask, lstm_encoders, forward_only, training)
    return results
