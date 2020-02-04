import pickle

import tensorflow as tf

from detext.model.bert import modeling

# the variable name in BERT
BERT_WORD_EMBEDDING_NAME = 'bert/embeddings/word_embeddings'


def init_word_embedding(hparams, mode, name_prefix="w"):
    """Initialize word embeddings from bert checkpoint, random initialization or pretrained word embedding."""
    embedding_name = "{}_embedding".format(name_prefix)
    if hparams.bert_checkpoint is not None:
        # Initializes word embedding from bert checkpoint
        embedding = tf.compat.v1.get_variable(BERT_WORD_EMBEDDING_NAME, [hparams.vocab_size, hparams.num_units],
                                              dtype=tf.float32, trainable=hparams.we_trainable)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = tf.trainable_variables()
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(
                tvars,
                hparams.bert_checkpoint)
            tf.train.init_from_checkpoint(hparams.bert_checkpoint, assignment_map)
            print("*** {} variables are initialized from bert pretrained model ****".format(
                len(initialized_variable_names)))
            print(initialized_variable_names)
        embedding = tf.Variable(embedding.initialized_value(), trainable=hparams.we_trainable, name=embedding_name)
    elif hparams.we_file is None:
        # Random initialization
        embedding = tf.compat.v1.get_variable(
            embedding_name, [hparams.vocab_size, hparams.num_units], dtype=tf.float32, trainable=hparams.we_trainable)
    else:
        # Initialize by pretrained word embedding
        print('mode=' + str(mode))
        print('Loading pretrained word embedding from {}'.format(hparams.we_file))
        we = pickle.load(tf.gfile.Open(hparams.we_file, 'rb'))
        assert hparams.vocab_size == we.shape[0] and hparams.num_units == we.shape[1]
        embedding = tf.compat.v1.get_variable(
            name=embedding_name,
            shape=[hparams.vocab_size, hparams.num_units],
            dtype=tf.float32,
            initializer=tf.constant_initializer(we),
            trainable=hparams.we_trainable
        )
    return embedding
