import tensorflow as tf

from detext.utils.model_utils import create_rnn_cell, init_word_embedding


class LstmModel():
    def __init__(self,
                 query,
                 doc_fields,
                 usr_fields,
                 hparams,
                 mode):
        """
        Applies LSTM to convert text to a fix length embedding

        :param query: Tensor(dtype=tf.int)  Shape=[batch_size, query_length]
        :param doc_fields: list(Tensor(dtype=int))/Tensor  A list of document fields. Each has shape=
            [batch_size, max_group_size, doc_field_length]. For online scoring, these fields may be precomputed and
            input as Tensor
        :param usr_fields: list(Tensor(dtype=int))/Tensor  A list of user fields. Each has shape=
            [batch_size, usr_field_length]. For online scoring, these fields may be precomputed and
            input as Tensor
        :param hparams: HParams
        :param mode: TRAIN/EVAL/INFER
        """
        self._query = query
        self._doc_fields = doc_fields
        self._usr_fields = usr_fields
        self._hparams = hparams
        self._mode = mode
        self.text_ftr_size = hparams.num_units

        # Initialize embedding
        self.embedding = init_word_embedding(self._hparams, self._mode)

        with tf.variable_scope("lstm", dtype=tf.float32):
            # RNN cell setup
            self.cell_dtype = tf.float32
            self.cell = create_rnn_cell(
                unit_type=hparams.unit_type,
                num_units=hparams.num_units,
                num_layers=hparams.num_layers,
                num_residual_layers=hparams.num_residual_layers,
                forget_bias=hparams.forget_bias,
                dropout=hparams.rnn_dropout,
                mode=tf.contrib.learn.ModeKeys.TRAIN,
            )
            if hparams.bidirectional:
                self.cell_fw = create_rnn_cell(
                    unit_type=hparams.unit_type,
                    num_units=hparams.num_units / 2,
                    num_layers=hparams.num_layers,
                    num_residual_layers=hparams.num_residual_layers,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.rnn_dropout,
                    mode=tf.contrib.learn.ModeKeys.TRAIN,
                )
                self.cell_bw = create_rnn_cell(
                    unit_type=hparams.unit_type,
                    num_units=hparams.num_units / 2,
                    num_layers=hparams.num_layers,
                    num_residual_layers=hparams.num_residual_layers,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.rnn_dropout,
                    mode=tf.contrib.learn.ModeKeys.TRAIN,
                )

            # Apply LSTM on text fields
            self.query_ftrs = self.apply_lstm_on_query() if query is not None else None
            self.usr_ftrs = self.apply_lstm_on_usr() if usr_fields is not None else None
            self.doc_ftrs = self.apply_lstm_on_doc()

    def apply_lstm_on_query(self):
        """
        Applies LSTM on query

        :return Tensor  Query features. Shape=[batch_size, text_ftr_size]
        """
        return self.apply_lstm_on_text(self._query)

    def apply_lstm_on_usr(self):
        """
        Applies LSTM on user fields

        :return Tensor  User features. Shape=[batch_size, num_usr_fields, text_ftr_size]
        """
        if type(self._usr_fields) is not list:
            return self._usr_fields

        usr_ftrs = []
        for i, usr_field in enumerate(self._usr_fields):
            usr_field_ftrs = self.apply_lstm_on_text(usr_field)
            usr_ftrs.append(usr_field_ftrs)
        usr_ftrs = tf.stack(usr_ftrs, axis=1)  # shape=[batch_size, num_usr_fields, text_ftr_size]
        return usr_ftrs

    def apply_lstm_on_doc(self):
        """Applies LSTM on documents

        :return Tensor  Document features. Shape=[batch_size, max_group_size, num_doc_fields, text_ftr_size]
        """

        # doc_fields should be the doc embeddings
        if type(self._doc_fields) is not list:
            return self._doc_fields

        doc_ftrs = []
        for i, doc_field in enumerate(self._doc_fields):
            doc_shape = tf.shape(doc_field)  # Shape=[batch_size, group_size, sent_length]
            doc_field = tf.reshape(doc_field, shape=[doc_shape[0] * doc_shape[1], doc_shape[2]])

            # Apply LSTM
            doc_field_ftrs = self.apply_lstm_on_text(doc_field)

            # Restore batch_size and group_size
            doc_field_ftrs = tf.reshape(doc_field_ftrs, shape=[doc_shape[0], doc_shape[1], self.text_ftr_size])
            doc_ftrs.append(doc_field_ftrs)
        doc_ftrs = tf.stack(doc_ftrs, axis=2)  # Shape=[batch_size, max_group_size, num_doc_fields, text_ftr_size]
        return doc_ftrs

    def apply_lstm_on_text(self, text):
        """
        Applies LSTM (language modeling) on text

        :param text: Tensor Shape=[batch_size, seq_len]
        :return Tensor Shape=[batch_size, num_units]
        """
        hparams = self._hparams

        # Shape=[batch_size]
        seq_len = tf.reduce_sum(tf.cast(tf.not_equal(text, hparams.pad_id), dtype=tf.int32), axis=-1)

        input_seq = text  # [batch_size, seq_len]
        input_emb = tf.nn.embedding_lookup(self.embedding, input_seq)  # [batch_size, seq_len, num_units]

        # Whether to use bidirectional RNN
        if hparams.bidirectional:
            _, final_states = tf.nn.bidirectional_dynamic_rnn(
                self.cell_fw,
                self.cell_bw,
                input_emb,
                dtype=self.cell_dtype,
                sequence_length=seq_len,
                time_major=False,
                swap_memory=True)
            if hparams.num_layers == 1:
                final_h = tf.concat([final_states[0].h, final_states[1].h], axis=1)
            else:
                final_h = tf.concat([final_states[0][-1].h, final_states[1][-1].h], axis=1)
        else:
            _, final_states = tf.nn.dynamic_rnn(
                self.cell,
                input_emb,
                dtype=self.cell_dtype,
                sequence_length=seq_len,
                time_major=False,
                swap_memory=True)  # [batch_size, seq_len, num_units]
            if hparams.num_layers == 1:
                final_h = final_states.h
            else:
                final_h = final_states[-1].h

        return final_h
