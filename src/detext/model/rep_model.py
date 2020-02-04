"""
Representation-based models.
"""

from functools import partial

import tensorflow as tf

from detext.model import bert_model, cnn_model, lstm_lm_model


class RepModel:
    """
    The representation-based model to generate deep features, based on the textual inputs: query, user fields,
    document fields.
    """

    def __init__(self,
                 query,
                 doc_fields,
                 usr_fields,
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
        self._doc_fields = doc_fields
        self._hparams = hparams
        self._mode = mode
        self._usr_fields = usr_fields

        # If SIM_WIDE_SCORING, no need to apply deep
        # Apply CNN or BERT
        if hparams.ftr_ext == 'cnn':
            self.scoring_model = cnn_model.CnnModel(query, doc_fields, usr_fields, hparams, mode)
        elif hparams.ftr_ext == 'bert':
            self.scoring_model = bert_model.BertModel(query, doc_fields, usr_fields, hparams, mode)
        elif hparams.ftr_ext == 'lstm_lm':
            self.scoring_model = lstm_lm_model.LstmLmModel(query, doc_fields, usr_fields, hparams, mode)
        else:
            raise ValueError('Currently only support cnn/bert model.')

        # Assign a name to each text feature in the graph so that we can quickly locate them in the graph
        for field in ['query_ftrs', 'usr_ftrs', 'doc_ftrs']:
            field_tensor = getattr(self.scoring_model, field)
            if field_tensor is not None:
                setattr(self.scoring_model, field, tf.identity(field_tensor, name=field))

        self.num_sim_ftrs, self.sim_ftrs = self.compute_sim_ftrs()

    def compute_sim_ftrs(self):
        """
        Computes the interaction features between user/document features and query features.
        When no query and user fields, then return doc_ftrs directly (as concatenation)

        Supported interaction modes (hparams.emb_sim_func) are:
            1. inner: computes cosine similarity between user fields/query and each document field
                num_sim_ftrs=hparams.num_doc_fields
            2. concat: concatenates user fields/query and each document field
                num_sim_ftrs=self.scoring_model.text_ftr_size * (num_usr_fields + hparams.num_doc_fields)
            3. hadamard: computes Hadamard product between user fields/query and each document field
                num_sim_ftrs=self.scoring_model.text_ftr_size * hparams.num_doc_fields * num_usr_fields

        :return num_sim_ftrs,sim_ftrs  Shape of sim_ftrs is [batch_size, group_size, num_sim_ftrs]. The value of
            num_sim_ftrs is as shown in the instructions above
        """
        hparams = self._hparams

        batch_size = tf.shape(self._doc_fields[0])[0]
        max_group_size = tf.shape(self._doc_fields[0])[1]

        query_ftrs = self.scoring_model.query_ftrs  # [batch_size, text_ftr_size]
        usr_ftrs = self.scoring_model.usr_ftrs  # [batch_size, num_usr_fields, text_ftr_size]
        doc_ftrs = self.scoring_model.doc_ftrs  # [batch_size, group_size, num_doc_fields, text_ftr_size]

        # If query is None, use doc_ftrs only and no interaction is performed
        if query_ftrs is None and usr_ftrs is None:
            sim_ftrs = tf.reshape(doc_ftrs, shape=[
                batch_size, max_group_size, self.scoring_model.text_ftr_size * hparams.num_doc_fields])
            num_sim_ftrs = self.scoring_model.text_ftr_size * hparams.num_doc_fields
            return num_sim_ftrs, sim_ftrs

        # Compute interaction between user text/query and document text
        num_usr_fields = 0
        tot_usr_ftrs = []
        if usr_ftrs is not None:
            tot_usr_ftrs.append(usr_ftrs)
            num_usr_fields += hparams.num_usr_fields

        # Treat query as one user field and append it to user field
        if query_ftrs is not None:
            query_ftrs = tf.expand_dims(query_ftrs, axis=1)  # [batch_size, 1, text_ftr_size]
            tot_usr_ftrs.append(query_ftrs)
            num_usr_fields += 1
        usr_ftrs = tf.concat(tot_usr_ftrs, axis=1)  # [batch_size, num_usr_fields, text_ftr_size]

        sim_ftrs, num_sim_ftrs = self.compute_sim_ftrs_for_usr_doc(doc_ftrs, usr_ftrs, hparams.num_doc_fields,
                                                                   num_usr_fields,
                                                                   hparams.emb_sim_func)
        return num_sim_ftrs, sim_ftrs

    def compute_sim_ftrs_for_usr_doc(self, doc_ftrs, usr_ftrs, hparams, num_usr_fields, sim_func_names):
        """Computes similarity between usr_ftrs and doc_ftrs with all function names specified in sim_func_names and
            concatenates the results

        :param doc_ftrs Shape=[batch_size, group_size, num_doc_fields, text_ftr_size]
        :param usr_ftrs Shape=[batch_size, num_usr_fields, text_ftr_size]
        :param sim_func_names Similarity function names
        """
        sim_ftrs_list = list()
        num_sim_ftrs_list = list()

        name2func = {'inner': compute_inner_ftrs_for_usr_doc,
                     'concat': compute_concat_ftrs_for_usr_doc,
                     'hadamard': compute_hadamard_prod_for_usr_doc}

        for name in sim_func_names:
            sim_ftrs, num_sim_ftrs = name2func[name](doc_ftrs, usr_ftrs, hparams, num_usr_fields,
                                                     self.scoring_model.text_ftr_size)
            sim_ftrs_list.append(sim_ftrs)
            num_sim_ftrs_list.append(num_sim_ftrs)

        return tf.concat(sim_ftrs_list, axis=-1), sum(num_sim_ftrs_list)


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
