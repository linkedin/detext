"""
The deep match model.
"""

import tensorflow as tf

from detext.model import rep_model
from detext.model.sp_linear_model import SparseLinearModel


class DeepMatch:
    """
    The deep match model
    """

    def __init__(self,
                 query,
                 wide_ftrs,
                 doc_fields,
                 hparams,
                 mode,
                 wide_ftrs_sp_idx=None,
                 wide_ftrs_sp_val=None,
                 usr_fields=None):
        """
        Build the deep match graph.
        :param query: Tensor  Input query. Shape=[batch_size, query_length]
        :param wide_ftrs: Tensor  Wide features. Shape=[batch_size, max_group_size, num_wide_ftrs]
        :param wide_ftrs_sp_idx: Tensor  Sparse wide features indices. Shape=[batch_size, max_group_size, num_wide_ftrs]
        :param wide_ftrs_sp_val: Tensor  Sparse wide features values. Shape=[batch_size, max_group_size, num_wide_ftrs]
        :param doc_fields: list(Tensor) or a pre-computed Tensor  List of document fields. Each has
            shape=[batch_size, max_group_size, doc_field_length]
        :param usr_fields: list(Tensor) or a pre-computed Tensor  List of user fields. Each has
            shape=[batch_size, usr_field_length]
        :param hparams: HParams  Hyper parameters
        :param mode: TRAIN/EVAL/INFER
        """
        self._query = query
        self._wide_ftrs = tf.where(tf.math.is_nan(wide_ftrs), tf.zeros_like(wide_ftrs),
                                   wide_ftrs) if wide_ftrs is not None else wide_ftrs
        self._wide_ftrs_sp_idx = wide_ftrs_sp_idx
        self._wide_ftrs_sp_val = wide_ftrs_sp_val
        self._doc_fields = doc_fields
        self._hparams = hparams
        self._mode = mode
        self._usr_fields = usr_fields

        self.batch_size, self.max_group_size = self.get_data_size()

        if hparams.use_deep:
            # Apply representation-based models
            self.deep_ftr_model = rep_model.RepModel(
                self._query,
                self._doc_fields,
                self._usr_fields,
                self._hparams,
                self._mode
            )
            # Assign a name to each text feature in the graph so that we can quickly locate them in the graph
            for field in ['query_ftrs', 'usr_ftrs', 'doc_ftrs']:
                field_tensor = getattr(self.deep_ftr_model.scoring_model, field)
                if field_tensor is not None:
                    setattr(self.deep_ftr_model.scoring_model, field, tf.identity(field_tensor, name=field))

        self.scores = self.compute_total_scores()  # score(wide&deep) + score(wide_sp)
        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self.scores = tf.transpose(self.scores, name='final_scores')  # final_scores is transposed

    def get_data_size(self):
        """get batch size, max group size"""
        if self._wide_ftrs is not None:
            data_shape = tf.shape(self._wide_ftrs)
        elif self._wide_ftrs_sp_idx is not None:
            data_shape = tf.shape(self._wide_ftrs_sp_idx)
        elif self._doc_fields is not None:
            data_shape = tf.shape(self._doc_fields[0])
        elif self._query is not None:
            data_shape = tf.shape(self._query)
            return data_shape[0], None
        else:
            raise ValueError('Cannot infer data size.')
        batch_size = data_shape[0]
        max_group_size = data_shape[1]
        return batch_size, max_group_size

    def compute_total_scores(self):
        """Computes total scores """
        scores = 0.
        hparams = self._hparams
        if hparams.use_deep or self._wide_ftrs is not None:
            scores += self.compute_score()
        if self._wide_ftrs_sp_idx is not None:
            linear_model, wide_sp_scores = self.compute_sp_wide_score()
            scores += wide_sp_scores
        return scores

    def compute_sp_wide_score(self):
        """Computes linear score from wide sparse features"""
        linear_model = SparseLinearModel(wide_ftrs_sp_idx=self._wide_ftrs_sp_idx,
                                         num_wide_sp=self._hparams.num_wide_sp,
                                         wide_ftrs_sp_val=self._wide_ftrs_sp_val)
        return linear_model, linear_model.score

    def compute_score(self):
        """
        Compute the score for a query/doc pair based on dense features and deep features
        :return Tensor  Matching score. Shape=[batch_size, max_group_size]
        """
        hparams = self._hparams
        batch_size = self.batch_size
        max_group_size = self.max_group_size

        num_sim_ftrs = hparams.num_doc_fields
        if hparams.use_deep:
            num_sim_ftrs, sim_ftrs = self.deep_ftr_model.num_sim_ftrs, self.deep_ftr_model.sim_ftrs

        if hparams.get('ftr_mean') is not None:
            print("--- use feature normalization ---")
            ftr_mean = tf.constant(hparams.ftr_mean, dtype=tf.float32)
            ftr_std = tf.constant(hparams.ftr_std, dtype=tf.float32)
            self._wide_ftrs = (self._wide_ftrs - ftr_mean) / ftr_std

        # elementwise rescaling for dense features
        if hparams.elem_rescale and self._wide_ftrs is not None:
            self.elem_norm_w = tf.compat.v1.get_variable("wide_ftr_norm_w", [hparams.num_wide], dtype=tf.float32,
                                                         initializer=tf.constant_initializer(1.0))
            self.elem_norm_b = tf.compat.v1.get_variable("wide_ftr_norm_b", [hparams.num_wide], dtype=tf.float32,
                                                         initializer=tf.constant_initializer(0.0))
            self._wide_ftrs = tf.tanh(self._wide_ftrs * self.elem_norm_w + self.elem_norm_b)

        emb_size = 0
        final_ftrs = []

        # whether to include deep features
        if hparams.use_deep:
            final_ftrs.append(sim_ftrs)
            emb_size += num_sim_ftrs

        # whether to include wide features
        if self._wide_ftrs is not None:
            final_ftrs.append(self._wide_ftrs)
            emb_size += hparams.num_wide

        # add whether empty
        if hparams.explicit_empty:
            ftrs, size = self.add_empty_str_ftr(max_group_size)
            final_ftrs += ftrs
            emb_size += size

        # reshape
        final_ftrs = tf.concat(final_ftrs, axis=-1)
        self.final_ftrs = tf.reshape(final_ftrs, shape=[batch_size, max_group_size, emb_size])

        # hidden layer
        hidden_ftrs = self.final_ftrs
        for i, hidden_size in enumerate(hparams.num_hidden):
            if hidden_size == 0:
                continue
            hidden_ftrs = tf.layers.dense(
                hidden_ftrs,
                units=hidden_size,
                use_bias=True,
                activation=tf.tanh,
                name="hidden_projection_" + str(i))
        # final score for query/doc pairs
        scores = tf.layers.dense(
            hidden_ftrs,
            units=1,
            use_bias=True,
            activation=None,
            name="scoring")
        scores = tf.squeeze(scores, axis=-1)
        return scores  # shape=[batch_size, max_group_size]

    def add_empty_str_ftr(self, max_group_size):
        """
        Add whether-the-string-is-empty features into final_ftrs.
        """
        num_ftrs = 0
        all_ftrs = []
        hparams = self._hparams

        def process_text(text, is_doc=False):
            # text length
            text_length = tf.reduce_sum(tf.cast(tf.not_equal(text, hparams.pad_id), dtype=tf.int32), axis=-1)
            # whether the text is empty
            text_is_empty = tf.cast(tf.equal(text_length, max(hparams.filter_window_sizes) * 2 - 2), dtype=tf.float32)
            if not is_doc:
                # for non-doc, the shape is [batch_size]
                text_is_empty = tf.expand_dims(tf.expand_dims(text_is_empty, axis=-1), axis=-1)
                text_is_empty = tf.tile(text_is_empty, [1, max_group_size, 1])
            else:
                # for doc, the shape is [batch_size, max_group_size]
                text_is_empty = tf.expand_dims(text_is_empty, axis=-1)
            all_ftrs.append(text_is_empty)

        if self._query is not None:
            process_text(self._query)
            num_ftrs += 1
        if self._doc_fields is not None:
            for doc_field in self._doc_fields:
                process_text(doc_field, True)
                num_ftrs += 1
        if self._usr_fields is not None:
            for usr_field in self._usr_fields:
                process_text(usr_field)
                num_ftrs += 1
        return all_ftrs, num_ftrs
