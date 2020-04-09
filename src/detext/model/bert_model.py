"""
The model that uses BERT to extract features from texts, and generate a score for a query/doc pair
Different from CNN, there is only one BERT model that generates the text embedding for all fields.
"""

import tensorflow as tf

from detext.bert_embedding_extraction.bert_utils import create_bert_model, init_from_checkpoint


class BertModel(object):
    """Apply BERT to convert text to a fix length embedding."""

    def __init__(self,
                 query,
                 doc_fields,
                 usr_fields,
                 hparams,
                 mode):
        """
        Apply BERT to convert text to features
        :param query: Shape=[batch_size, query_length]
        :param doc_fields: list(Tensor) or a pre-computed Tensor  List of document fields. Each has
            shape=[batch_size, max_group_size, doc_field_length]
        :param usr_fields: list(Tensor) or a pre-computed Tensor  List of user fields. Each has
            shape=[batch_size, usr_field_length]
        :param hparams:
        :param mode:
        """
        self.query = query
        self.doc_fields = doc_fields
        self.usr_fields = usr_fields
        self.hparams = hparams
        self.mode = mode

        self.query_ftrs, self.doc_ftrs, self.usr_ftrs = self.apply_bert()

    def apply_bert(self):
        """Sends the query and doc_fields to BERT to get text embedding.

        If query is None, the returned query_ftrs is also None
        :return query_ftrs,doc_ftrs,usr_ftrs  shape(query_ftrs)=[batch_size, num_units],
            shape(doc_ftrs)=[batch_size, max_group_size, num_doc_fields, num_units],
            shape(usr_ftrs)=[batch_size, num_doc_fields, num_units]
        """
        hparams = self.hparams
        query = self.query
        has_query = query is not None
        has_doc = self.doc_fields is not None

        # at least one of query and doc_fields should be true
        if (not has_query) and (not has_doc):
            raise ValueError('query and doc_fields cannot both be None')

        doc_fields = self.doc_fields
        usr_fields = self.usr_fields

        # if doc is None, we cannot infer batch_size and max_group_size from doc
        batch_size = tf.shape(doc_fields[0])[0] if has_doc else tf.shape(query)[0]
        max_group_size = tf.shape(doc_fields[0])[1] if has_doc else None

        if not has_query:
            # Instead of removing the query sequence in the input id array fed into BERT, we use a fake query with
            #   the same size as the first document field in the first document. The content of this query will NOT
            #   be used anywhere. It is just to save effort on code changes handling the case when `query is None`
            query = tf.ones(dtype=tf.int32, shape=tf.shape(doc_fields[0][:, 0, :]))

        # 1. combine query and doc_fields into one matrix
        # find largest text length value
        max_text_len = tf.shape(query)[-1]
        max_text_len_array = [max_text_len]

        # Add doc field length info
        if type(doc_fields) is list:
            for doc_field in doc_fields:
                doc_field_text_len = tf.shape(doc_field)[-1]
                max_text_len_array.append(doc_field_text_len)
                max_text_len = tf.maximum(max_text_len, doc_field_text_len)

        # Add usr field length info
        if type(usr_fields) is list:
            for usr_field in usr_fields:
                usr_field_text_len = tf.shape(usr_field)[-1]
                max_text_len_array.append(usr_field_text_len)
                max_text_len = tf.maximum(max_text_len, usr_field_text_len)

        # pad data
        padded_query = tf.pad(query, [[0, 0], [0, max_text_len - max_text_len_array[0]]],
                              constant_values=hparams.pad_id)
        bert_input_ids_array = [padded_query]

        if type(doc_fields) is list:
            for doc_field, doc_field_max_text_len in zip(doc_fields, max_text_len_array[1:]):
                # Shape=[batch_size*max_group_size, doc_field_max_text_len]
                doc_field_2d = tf.reshape(doc_field, shape=[-1, doc_field_max_text_len])
                padded_doc_field = tf.pad(
                    doc_field_2d, paddings=[[0, 0], [0, max_text_len - doc_field_max_text_len]],
                    constant_values=hparams.pad_id)
                bert_input_ids_array.append(padded_doc_field)

        if type(usr_fields) is list:
            doc_field_end = 1 + len(doc_fields) if type(doc_fields) is list else 1

            for usr_field, usr_field_max_text_len in zip(usr_fields, max_text_len_array[doc_field_end:]):
                # Shape=[batch_size, usr_field_max_text_len]
                padded_usr_field = tf.pad(usr_field, paddings=[[0, 0], [0, max_text_len - usr_field_max_text_len]],
                                          constant_values=hparams.pad_id)
                bert_input_ids_array.append(padded_usr_field)

        self.bert_input_ids = tf.concat(bert_input_ids_array, axis=0)
        self.bert_input_mask = tf.cast(tf.not_equal(self.bert_input_ids, hparams.pad_id), dtype=tf.int32)

        # 2. apply bert
        self.text_ftr_size = hparams.num_units

        bertm = create_bert_model(
            bert_config_path=hparams.bert_config_file,
            is_training=False,
            input_ids=self.bert_input_ids,
            input_mask=self.bert_input_mask,
            use_one_hot_embeddings=False
        )

        # initialize bert with pretrained checkpoint
        if self.mode == tf.estimator.ModeKeys.TRAIN and self.hparams.bert_checkpoint is not None:
            init_from_checkpoint(self.hparams.bert_checkpoint)

        # 3. get query embedding and doc_fields embedding
        last_layer = tf.squeeze(bertm.get_all_encoder_layers()[-1][:, 0:1, :], axis=1)
        pooled_output = tf.layers.dense(
            last_layer,
            self.text_ftr_size,
            activation=tf.tanh,
            name='bert/last_layer_cls'
        )  # shape = [# of texts, bert_dim]

        # dropout
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output = tf.nn.dropout(pooled_output, keep_prob=0.9)

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
        if type(usr_fields) is not list:
            usr_ftrs = self.usr_fields
        else:
            usr_field_start = doc_field_end
            usr_ftrs = []
            for _ in range(len(usr_fields)):
                usr_field_end = usr_field_start + batch_size
                uftrs = pooled_output[usr_field_start: usr_field_end]
                usr_ftrs.append(uftrs)
                usr_field_start = usr_field_end
            usr_ftrs = tf.stack(usr_ftrs, axis=1)  # shape=[batch_size, num_usr_fields, num_units]

        return query_ftrs, doc_ftrs, usr_ftrs
