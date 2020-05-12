"""
Representation-based models.
"""
import tensorflow as tf
from functools import partial

from detext.model import bert_model
from detext.model import cnn_model
from detext.model import id_embed_model
from detext.model import lstm_lm_model
from detext.model import lstm_model


class RepModel(object):
    """
    The representation-based model to generate deep features, based on the textual and id inputs: query, user fields,
    document fields.

    Attributes:
        doc_ftrs  Tensor Shape=[batch_size, max_group_size, num_doc_fields, ftr_size]
        usr_ftrs  Tensor Shape=[batch_size, num_usr_fields, ftr_size]
        num_doc_fields int Number of document fields (text + id)
        num_usr_fields int Number of document fields (text + id)
        num_sim_ftrs  int Number of similarity features
        sim_ftrs  Tensor Shape=[batch_size, max_group_size, num_sim_ftrs] Similarity features
    """

    def __init__(self,
                 query,
                 doc_fields,
                 usr_fields,
                 doc_id_fields,
                 usr_id_fields,
                 hparams,
                 mode):
        """
        :param query: Tensor  Input query. Shape=[batch_size, query_length]
        :param doc_fields: list(Tensor) or a pre-computed Tensor  List of document fields. Each has
            shape=[batch_size, max_group_size, doc_field_length]
        :param usr_fields: list(Tensor) or a pre-computed Tensor  List of user fields. Each has
            shape=[batch_size, usr_field_length]
        :param hparams: HParams  Hyper parameters
        :param mode: TRAIN/EVAL/INFER
        """
        self._query = query
        self._hparams = hparams
        self._mode = mode
        self._usr_fields = usr_fields
        self._doc_fields = doc_fields
        self._usr_id_fields = usr_id_fields
        self._doc_id_fields = doc_id_fields

        # Apply text encoder on query/doc/user fields
        ftr_ext_name2model = {
            'cnn': cnn_model.CnnModel,
            'bert': bert_model.BertModel,
            'lstm_lm': lstm_lm_model.LstmLmModel,
            'lstm': lstm_model.LstmModel
        }
        self.text_encoding_model = ftr_ext_name2model[hparams.ftr_ext](query, doc_fields, usr_fields, hparams, mode)

        # Apply embedding lookup and averaging for usr/doc id features
        self.id_encoding_model = id_embed_model.IdEmbedModel(
            doc_id_fields=self._doc_id_fields,
            usr_id_fields=self._usr_id_fields,
            hparams=self._hparams,
            mode=self._mode
        )

        # Use text_ftr_size as the size of representation for id embedding and text embedding
        self.ftr_size = self.text_encoding_model.text_ftr_size

        # Add text ftrs and id ftrs together
        for field in ['query_ftrs', 'usr_ftrs', 'doc_ftrs']:
            text_field_tensor = getattr(self.text_encoding_model, field, None)
            id_field_tensor = getattr(self.id_encoding_model, field, None)

            # Project text embedding to representation space
            # if text_field_tensor is not None:
            #     text_field_tensor = tf.layers.dense(text_field_tensor, units=self.ftr_size, use_bias=True,
            #                                         name="text_projection", reuse=tf.AUTO_REUSE)

            # Project id embedding to representation space
            if id_field_tensor is not None:
                id_field_tensor = tf.layers.dense(id_field_tensor, units=self.ftr_size, use_bias=True,
                                                  name="id_projection", reuse=tf.AUTO_REUSE)

            # Compose a new tensor to include user id info and user text info
            # I.e., if there are 2 user id fields, 3 user text fields, we have 2 + 3 = 5 user embedding fields
            # Same processing also works for doc fields
            if text_field_tensor is not None and id_field_tensor is not None:
                composite_field_tensor = tf.concat([text_field_tensor, id_field_tensor], axis=-2)
            elif text_field_tensor is not None:
                composite_field_tensor = text_field_tensor
            else:
                composite_field_tensor = id_field_tensor

            # Assign a name to each text feature in the graph so that we can quickly locate them in the graph
            if composite_field_tensor is not None:
                composite_field_tensor = tf.identity(composite_field_tensor, name=field)
            setattr(self, field, composite_field_tensor)

        self.num_doc_fields = hparams.num_doc_fields + hparams.num_doc_id_fields
        self.num_usr_fields = hparams.num_usr_fields + hparams.num_usr_id_fields

        self.num_sim_ftrs, self.sim_ftrs = self.compute_sim_ftrs()

    def get_data_size(self):
        """Infers batch size, max group size"""
        if self.doc_ftrs is not None:
            data = self.doc_ftrs
        elif self.query_ftrs:
            data = self.query_ftrs
        elif self.usr_ftrs:
            data = self.usr_ftrs
        else:
            raise ValueError('Cannot infer data size.')
        data_shape = tf.shape(data)
        return data_shape[0], data_shape[1]

    def compute_sim_ftrs(self):
        """
        Computes the interaction features between user/document features and query features.
        When no query and user fields, then return doc_ftrs directly (as concatenation)

        Supported interaction modes (hparams.emb_sim_func) are:
            1. inner: computes cosine similarity between user fields/query and each document field
                num_sim_ftrs=num_doc_fields
            2. concat: concatenates user fields/query and each document field
                num_sim_ftrs=self.ftr_size * (num_usr_fields + num_doc_fields)
            3. hadamard: computes Hadamard product between user fields/query and each document field
                num_sim_ftrs=self.ftr_size * num_doc_fields * num_usr_fields

        :return num_sim_ftrs,sim_ftrs  Shape of sim_ftrs is [batch_size, group_size, num_sim_ftrs]. The value of
            num_sim_ftrs is as shown in the instructions above
        """
        hparams = self._hparams

        batch_size, max_group_size = self.get_data_size()
        num_doc_fields = self.num_doc_fields
        ftr_size = self.ftr_size

        query_ftrs = self.query_ftrs  # [batch_size, ftr_size]
        usr_ftrs = self.usr_ftrs  # [batch_size, num_usr_fields, ftr_size]
        doc_ftrs = self.doc_ftrs  # [batch_size, group_size, num_doc_fields, ftr_size]

        # If query is None, use doc_ftrs only and no interaction is performed
        if query_ftrs is None and usr_ftrs is None:
            sim_ftrs = tf.reshape(doc_ftrs, shape=[batch_size, max_group_size, ftr_size * num_doc_fields])
            num_sim_ftrs = ftr_size * num_doc_fields
            return num_sim_ftrs, sim_ftrs

        # If use_doc_projection, then the n doc fields are projected to 1 vector space
        if hparams.use_doc_projection:
            doc_ftrs = tf.reshape(doc_ftrs, shape=[batch_size, max_group_size, 1, ftr_size * num_doc_fields])
            doc_ftrs = tf.layers.dense(doc_ftrs,
                                       ftr_size,
                                       use_bias=True,
                                       activation=tf.tanh,
                                       name="doc_ftrs_projection_layer")  # [batch_size, max_group_size, 1, ftr_size]
            doc_ftrs = tf.identity(doc_ftrs, name='doc_ftrs_projection')
            num_doc_fields = 1

        # Compute interaction between user text/query and document text
        num_usr_fields = 0
        tot_usr_ftrs = []
        if usr_ftrs is not None:
            # If use_usr_projection, then the n usr fields are projected to 1 vector space
            if hparams.use_usr_projection:
                usr_ftrs = tf.reshape(usr_ftrs, shape=[batch_size, 1, ftr_size * self.num_usr_fields])
                usr_ftrs = tf.layers.dense(usr_ftrs,
                                           ftr_size,
                                           use_bias=True,
                                           activation=tf.tanh,
                                           name="usr_ftrs_projection_layer")  # [batch_size, 1, ftr_size]
                usr_ftrs = tf.identity(usr_ftrs, name='usr_ftrs_projection')
                num_usr_fields = 1
            else:
                num_usr_fields += self.num_usr_fields
            tot_usr_ftrs.append(usr_ftrs)

        # Treat query as one user field and append it to user field
        if query_ftrs is not None:
            query_ftrs = tf.expand_dims(query_ftrs, axis=1)  # [batch_size, 1, ftr_size]
            tot_usr_ftrs.append(query_ftrs)
            num_usr_fields += 1
        usr_ftrs = tf.concat(tot_usr_ftrs, axis=1)  # [batch_size, num_usr_fields, ftr_size]

        sim_ftrs, num_sim_ftrs = self.compute_sim_ftrs_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields,
                                                                   num_usr_fields,
                                                                   hparams.emb_sim_func)
        return num_sim_ftrs, sim_ftrs

    def compute_sim_ftrs_for_usr_doc(self, doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields, sim_func_names):
        """Computes similarity between usr_ftrs and doc_ftrs with all function names specified in sim_func_names and
            concatenates the results

        :param doc_ftrs Shape=[batch_size, group_size, num_doc_fields, ftr_size]
        :param usr_ftrs Shape=[batch_size, num_usr_fields, ftr_size]
        :param sim_func_names Similarity function names
        """
        sim_ftrs_list = list()
        num_sim_ftrs_list = list()

        name2func = {'inner': compute_inner_ftrs_for_usr_doc,
                     'concat': compute_concat_ftrs_for_usr_doc,
                     'hadamard': compute_hadamard_prod_for_usr_doc,
                     'diff': compute_elem_diff_for_usr_doc}

        for name in sim_func_names:
            sim_ftrs, num_sim_ftrs = name2func[name](doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields,
                                                     self.ftr_size)
            sim_ftrs_list.append(sim_ftrs)
            num_sim_ftrs_list.append(num_sim_ftrs)

        return tf.concat(sim_ftrs_list, axis=-1), sum(num_sim_ftrs_list)


def compute_elem_diff_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields, num_deep):
    """Computes Elementwise difference between usr_field_ftrs and doc_ftrs

    :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param usr_field_ftrs Shape=[batch_size, num_usr_fields, num_deep]
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features (the last dimension
    of sim_ftrs)
    """

    def compute_diff(usr_field_ftrs, doc_ftrs):
        """ Computes Elementwise diff between usr_field_ftrs and doc_ftrs

        :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
        :param usr_field_ftrs Shape=[batch_size, num_deep]
        """
        usr_field_ftrs = tf.expand_dims(tf.expand_dims(usr_field_ftrs, axis=1), axis=1)  # [batch_size, 1, 1, num_deep]
        elementwise_diff = usr_field_ftrs - doc_ftrs  # [batch_size, group_size, num_doc_fields, num_deep]
        return elementwise_diff  # [batch_size, group_size, num_doc_fields, num_deep]

    batch_size = tf.shape(doc_ftrs)[0]
    max_group_size = tf.shape(doc_ftrs)[1]

    # Shape=[num_usr_fields, batch_size, group_size, num_doc_fields, num_deep]
    sim_ftrs = tf.map_fn(partial(compute_diff, doc_ftrs=doc_ftrs), tf.transpose(usr_ftrs, [1, 0, 2]))
    # Shape=[batch_size, group_size, num_doc_fields, num_usr_fields, num_deep]
    sim_ftrs = tf.transpose(sim_ftrs, [1, 2, 3, 4, 0])

    num_sim_ftrs = num_doc_fields * num_usr_fields * num_deep
    # Shape=[batch_size, group_size, num_sim_ftrs]
    sim_ftrs = tf.reshape(sim_ftrs, [batch_size, max_group_size, num_sim_ftrs])
    return sim_ftrs, num_sim_ftrs


def compute_hadamard_prod_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields, num_deep):
    """Computes Hadamard product between usr_field_ftrs and doc_ftrs

    :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param usr_field_ftrs Shape=[batch_size, num_usr_fields, num_deep]
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features (the last dimension
    of sim_ftrs)
    """

    def compute_hadamard(usr_field_ftrs, doc_ftrs):
        """ Computes Hadamard product between usr_field_ftrs and doc_ftrs

        :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
        :param usr_field_ftrs Shape=[batch_size, num_deep]
        """
        usr_field_ftrs = tf.expand_dims(tf.expand_dims(usr_field_ftrs, axis=1), axis=1)  # [batch_size, 1, 1, num_deep]
        hadamard_prod = usr_field_ftrs * doc_ftrs  # [batch_size, group_size, num_doc_fields, num_deep]
        return hadamard_prod  # [batch_size, group_size, num_doc_fields, num_deep]

    batch_size = tf.shape(doc_ftrs)[0]
    max_group_size = tf.shape(doc_ftrs)[1]

    # Shape=[num_usr_fields, batch_size, group_size, num_doc_fields, num_deep]
    sim_ftrs = tf.map_fn(partial(compute_hadamard, doc_ftrs=doc_ftrs), tf.transpose(usr_ftrs, [1, 0, 2]))
    # Shape=[batch_size, group_size, num_doc_fields, num_usr_fields, num_deep]
    sim_ftrs = tf.transpose(sim_ftrs, [1, 2, 3, 4, 0])

    num_sim_ftrs = num_doc_fields * num_usr_fields * num_deep
    # Shape=[batch_size, group_size, num_sim_ftrs]
    sim_ftrs = tf.reshape(sim_ftrs, [batch_size, max_group_size, num_sim_ftrs])
    return sim_ftrs, num_sim_ftrs


def compute_concat_ftrs_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields, num_deep):
    """Concatenates doc_ftrs and usr_ftrs

    :param doc_ftrs Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param usr_ftrs Shape=[batch_size, num_usr_fields, num_deep]
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features (the last dimension
    of sim_ftrs)
    """
    batch_size = tf.shape(doc_ftrs)[0]
    max_group_size = tf.shape(doc_ftrs)[1]

    doc_ftrs = tf.reshape(doc_ftrs, shape=[batch_size, max_group_size, num_deep * num_doc_fields])

    usr_ftrs = tf.reshape(usr_ftrs, shape=[batch_size, 1, -1]) + tf.zeros([
        batch_size, max_group_size, num_deep * num_usr_fields])

    sim_ftrs = tf.concat([doc_ftrs, usr_ftrs], axis=2)
    num_sim_ftrs = num_deep * (num_doc_fields + num_usr_fields)
    return sim_ftrs, num_sim_ftrs


def compute_inner_ftrs_for_usr_doc(doc_ftrs, usr_ftrs, num_doc_fields, num_usr_fields, num_deep):
    """Computes cosine similarity between user and doc fields.

    :param doc_ftrs Shape=[batch_size, group_size, num_doc_fields, num_deep]
    :param usr_ftrs Shape=[batch_size, num_usr_fields, num_deep]
    :return (sim_ftrs, num_sim_ftrs). Similarity features and the number of similarity features, which is the last
        dimension of sim_ftrs
    """

    def compute_inner_sim(usr_field_ftrs, doc_ftrs):
        """Computes cosine similarity score between usr_field_ftrs and doc_ftrs

        :param doc_ftrs Document features. Shape=[batch_size, group_size, num_doc_fields, num_deep]
        :param usr_field_ftrs Shape=[batch_size, num_deep]
        """
        usr_field_ftrs = tf.expand_dims(tf.expand_dims(usr_field_ftrs, axis=1), axis=1)
        sim_ftrs = tf.reduce_sum(usr_field_ftrs * doc_ftrs, axis=-1)
        return sim_ftrs  # [batch_size, group_size, num_doc_fields]

    batch_size = tf.shape(doc_ftrs)[0]
    max_group_size = tf.shape(doc_ftrs)[1]

    doc_ftrs = tf.nn.l2_normalize(doc_ftrs, axis=-1)
    usr_ftrs = tf.nn.l2_normalize(usr_ftrs, axis=-1)
    # Shape=[num_usr_fields, batch_size, group_size, num_doc_fields]
    sim_ftrs = tf.map_fn(partial(compute_inner_sim, doc_ftrs=doc_ftrs), tf.transpose(usr_ftrs, [1, 0, 2]))
    # Shape=[batch_size, group_size, num_doc_fields, num_usr_fields]
    sim_ftrs = tf.transpose(sim_ftrs, [1, 2, 3, 0])
    num_sim_ftrs = num_doc_fields * num_usr_fields
    # Shape=[batch_size, group_size, num_doc_fields * num_usr_fields]
    sim_ftrs = tf.reshape(sim_ftrs, [batch_size, max_group_size, num_sim_ftrs])
    return sim_ftrs, num_sim_ftrs
