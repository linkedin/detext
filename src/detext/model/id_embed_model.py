import tensorflow as tf

from detext.utils.model_utils import init_word_embedding


class IdEmbedModel:
    """Model that produces id embedding for given doc/usr id fields

    Attributes:
        doc_ftrs  Tensor Shape=[batch_size, max_group_size, num_doc_id_fields, num_units_for_id_ftr]
        usr_ftrs  Tensor Shape=[batch_size, num_usr_id_fields, num_units_for_id_ftr]
    """

    def __init__(self,
                 doc_id_fields,
                 usr_id_fields,
                 hparams,
                 mode
                 ):
        """
        :param doc_id_fields: list(Tensor)  List of document fields. Each has
            shape=[batch_size, max_group_size, num_doc_id]
        :param usr_id_fields: list(Tensor)  List of user fields. Each has
            shape=[batch_size, num_usr_id]
        :param hparams: HParams  Hyper parameters
        :param mode: TRAIN/EVAL/INFER
        """
        self._usr_id_fields = usr_id_fields
        self._doc_id_fields = doc_id_fields
        self._hparams = hparams
        self._mode = mode
        self.id_ftr_size = hparams.num_units_for_id_ftr

        with tf.variable_scope("id_embed", dtype=tf.float32):
            if usr_id_fields is not None or usr_id_fields is not None:
                self.embedding = init_word_embedding(
                    {"we_trainable": hparams.we_trainable_for_id_ftr,
                     "we_file": hparams.we_file_for_id_ftr,
                     "vocab_size": hparams.vocab_size_for_id_ftr,
                     "num_units": hparams.num_units_for_id_ftr
                     }, self._mode)
            self.usr_ftrs = self.apply_embed_on_usr_id() if usr_id_fields is not None else None
            self.doc_ftrs = self.apply_embed_on_doc_id() if doc_id_fields is not None else None

    def apply_embed_on_usr_id(self):
        """Applies embedding lookup and averaging for user id features

        :return Tensor Shape=[batch_size, num_usr_id_fields, num_units_for_id_ftr]
        """
        hparams = self._hparams
        usr_ftrs = []
        for i, usr_field in enumerate(self._usr_id_fields):
            seq_mask = tf.cast(tf.not_equal(usr_field, hparams.pad_id_for_id_ftr),
                               dtype=tf.float32)  # [batch_size, num_usr_id]
            seq_mask = tf.expand_dims(seq_mask, axis=-1)  # [batch_size, num_usr_id, 1]
            seq_length = tf.reduce_sum(seq_mask, axis=-2)  # [batch_size, 1]

            usr_id_embeddings = tf.nn.embedding_lookup(
                self.embedding, usr_field)  # [batch_size, num_usr_id, num_units_for_id_ftr]
            sum_usr_id_embedding = tf.reduce_sum(
                usr_id_embeddings * seq_mask, axis=1)  # [batch_size, num_units_for_id_ftr]
            usr_id_avg_embedding = tf.div_no_nan(sum_usr_id_embedding, seq_length)  # [batch_size, num_units_for_id_ftr]
            usr_ftrs.append(usr_id_avg_embedding)
        return tf.stack(usr_ftrs, axis=1)

    def apply_embed_on_doc_id(self):
        """Applies embedding lookup and averaging for doc id features

        :return Tensor Shape=[batch_size, max_group_size, num_doc_id_fields, num_units_for_id_ftr]
        """
        hparams = self._hparams

        doc_ftrs = []
        for i, doc_field in enumerate(self._doc_id_fields):
            seq_mask = tf.cast(tf.not_equal(doc_field, hparams.pad_id_for_id_ftr),
                               dtype=tf.float32)  # [batch_size, max_group_size, num_doc_id]
            seq_mask = tf.expand_dims(seq_mask, axis=-1)  # [batch_size, max_group_size, num_doc_id, 1]
            seq_length = tf.reduce_sum(seq_mask, axis=-2,)  # [batch_size, max_group_size, 1]

            doc_id_embeddings = tf.nn.embedding_lookup(
                self.embedding, doc_field, )  # [batch_size, max_group_size, num_doc_id, num_units_for_id_ftr]
            sum_doc_id_embedding = tf.reduce_sum(
                doc_id_embeddings * seq_mask, axis=2)  # [batch_size, max_group_size, num_units_for_id_ftr]
            doc_id_avg_embedding = tf.div_no_nan(sum_doc_id_embedding,
                                                 seq_length)  # [batch_size, max_group_size, num_units_for_id_ftr]
            doc_ftrs.append(doc_id_avg_embedding)
        return tf.stack(doc_ftrs, axis=2)
