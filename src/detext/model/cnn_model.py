"""
The model that uses CNN to extract features from texts, and generate a score for a query/doc pair
"""

import tensorflow as tf

from detext.utils import model_utils


class CnnModel(object):

    def __init__(self,
                 query,
                 doc_fields,
                 usr_fields,
                 hparams,
                 mode):
        """
        Applies CNN to convert text to a fix length embedding

        :param query: Tensor(dtype=tf.int)  Shape=[batch_size, query_length]
        :param doc_fields: list(Tensor(dtype=int))/Tensor  A list of document fields. Each has shape=
            [batch_size, max_group_size, doc_field_length]. For online scoring, these fields may be precomputed and
            input as Tensor
        :param usr_fields: list(Tensor(dtype=int))/Tensor  A list of user fields. Each has shape=
            [batch_size, usr_field_length]. For online scoring, these fields may be precomputed and
            input as Tensor
        :param hparams: HParams  Important CNN params are: filter_window_sizes:list(int), num_filters:int
        :param mode: TRAIN/EVAL/INFER
        """
        self._query = query
        self._doc_fields = doc_fields
        self._usr_fields = usr_fields
        self._hparams = hparams
        self._mode = mode
        self.text_ftr_size = len(hparams.filter_window_sizes) * hparams.num_filters

        self.embedding = model_utils.init_word_embedding(self._hparams, self._mode)
        with tf.variable_scope("cnn", dtype=tf.float32):
            self.query_ftrs = self.apply_cnn_on_query() if query is not None else None
            self.usr_ftrs = self.apply_cnn_on_usr() if usr_fields is not None else None
            self.doc_ftrs = self.apply_cnn_on_doc()

    def apply_cnn_on_query(self):
        """
        Apply cnn on queries

        :return Tensor  Query features. Shape=[batch_size, num_filters*len(filter_window_size)]
        """
        hparams = self._hparams
        query_emb = tf.nn.embedding_lookup(self.embedding, self._query)  # shape=[batch_size, sent_length, num_units]
        query_length = tf.reduce_sum(tf.cast(tf.not_equal(self._query, hparams.pad_id), dtype=tf.int32), axis=-1)
        query_ftrs = apply_cnn_on_text(query_emb,
                                       hparams.filter_window_sizes,
                                       hparams.num_filters,
                                       hparams.num_units,
                                       'query_cnn_',
                                       query_length,
                                       set_empty_zero=hparams.explicit_empty)
        return query_ftrs

    def apply_cnn_on_usr(self):
        """
        Apply cnn on user fields

        :return Tensor  User features. Shape=[batch_size, num_usr_fields, num_filters*len(filter_window_size)]
        """
        if type(self._usr_fields) is not list:
            return self._usr_fields

        hparams = self._hparams
        usr_ftrs = []
        for i, usr_field in enumerate(self._usr_fields):
            usr_emb = tf.nn.embedding_lookup(self.embedding, usr_field)  # shape=[batch_size, sent_length, num_units]
            usr_length = tf.reduce_sum(tf.cast(tf.not_equal(usr_field, hparams.pad_id), dtype=tf.int32), axis=-1)
            # shape=[batch_size, num_filters*len(filter_window_sizes)]
            usr_field_ftrs = apply_cnn_on_text(usr_emb,
                                               hparams.filter_window_sizes,
                                               hparams.num_filters,
                                               hparams.num_units,
                                               'usr_cnn_{}'.format(i),
                                               usr_length,
                                               set_empty_zero=hparams.explicit_empty)
            usr_ftrs.append(usr_field_ftrs)
        usr_ftrs = tf.stack(usr_ftrs, axis=1)  # shape=[batch_size, num_usr_fields, num_filters*len(filter_window_size)]
        return usr_ftrs

    def apply_cnn_on_doc(self):
        """Apply cnn on documents

        :return Tensor  Document features. Shape=[batch_size, max_group_size, num_doc_fields, num_ftrs]
        """

        # doc_fields should be the doc embeddings
        if type(self._doc_fields) is not list:
            return self._doc_fields

        hparams = self._hparams
        doc_ftrs = []
        for i, doc_field in enumerate(self._doc_fields):
            doc_emb = tf.nn.embedding_lookup(self.embedding,
                                             doc_field)  # shape=[batch_size, group_size, sent_length, num_units]

            # reshape docs to merge batch_size and group_size
            doc_emb_shape = tf.shape(doc_emb)
            doc_emb_reshape = tf.reshape(
                doc_emb, shape=[doc_emb_shape[0] * doc_emb_shape[1], doc_emb_shape[2], doc_emb_shape[3]])
            doc_length = tf.reduce_sum(tf.cast(tf.not_equal(doc_field, hparams.pad_id), dtype=tf.int32), axis=-1)
            doc_length = tf.reshape(doc_length, shape=[doc_emb_shape[0] * doc_emb_shape[1]])
            # apply cnn
            doc_field_ftrs = apply_cnn_on_text(doc_emb_reshape,
                                               hparams.filter_window_sizes,
                                               hparams.num_filters,
                                               hparams.num_units,
                                               'doc_cnn_' + str(i) + '_',
                                               doc_length,
                                               set_empty_zero=hparams.explicit_empty)
            # restore batch_size and group_size
            doc_field_ftrs = tf.reshape(doc_field_ftrs, shape=[
                doc_emb_shape[0], doc_emb_shape[1], self.text_ftr_size])
            doc_ftrs.append(doc_field_ftrs)
        doc_ftrs = tf.stack(doc_ftrs, axis=2)  # shape=[batch_size, max_group_size, num_doc_fields, num_ftrs]
        return doc_ftrs


def apply_cnn_on_text(text_emb,
                      filter_window_sizes,
                      num_filters,
                      num_units,
                      name_prefix,
                      text_length=None,
                      set_empty_zero=False):
    """
    For any text, apply several cnn with the same window size.
    The final embedding is concatenated from the cnn outputs.

    :return Tensor  Text features. Shape=[batch_size, num_filters*len(filter_window_sizes)]
    """
    text_ftrs = []
    for i in filter_window_sizes:
        t_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(text_emb, 3),
            filters=num_filters,
            kernel_size=[i, num_units],
            activation=tf.nn.relu,
            padding='valid',
            name=name_prefix + str(i)
        )  # shape = [batch_size, q_len-filter_size+1, 1, num_filters]
        if text_length is not None:
            # Make sure the convolution results is consistent
            mask_length = tf.maximum(text_length - i + 1, 0)
            mask = tf.sequence_mask(mask_length, maxlen=tf.shape(text_emb)[1] - i + 1, dtype=tf.float32)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=2), axis=2)
            t_conv *= mask

        t_pool = tf.reduce_max(t_conv, axis=1)  # shape = [batch_size, 1, num_filters]
        t_pool = tf.squeeze(t_pool, axis=1)  # shape = [batch_size, num_filters]

        # set the cnn output of empty strings as 0
        if set_empty_zero:
            not_empty_str = tf.cast(tf.not_equal(text_length, max(filter_window_sizes) * 2 - 2), dtype=tf.float32)
            t_pool *= tf.expand_dims(not_empty_str, axis=1)

        text_ftrs.append(t_pool)
    text_ftrs = tf.concat(text_ftrs, 1)  # shape = [batch_size, num_filters*len(filter_window_sizes)]
    return text_ftrs
