import tensorflow as tf


def compute_regularization_penalty(l1, l2, trainable_vars):
    """ Returns the regularization penalty specified in hparams """
    l1 = l1 if l1 is not None else 0
    l2 = l2 if l2 is not None else 0
    regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    penalty = 0.0
    for weight in trainable_vars:
        penalty += regularizer(weight)
    return penalty


def compute_text_generation_loss(logits, labels, lengths):
    """ Returns categorical crossentropy for given sequence

    :param logits: Shape=[batch_size, max_sentence_length, vocab_size]
    :param labels: Shape=[batch_size, max_sentence_length]
    :param lengths: Shape=[batch_size]
    """
    loss_val = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True)  # [batch_size, max_sentence_length]

    mask = tf.sequence_mask(lengths, maxlen=tf.shape(labels)[1], dtype=tf.dtypes.float32)  # [batch_size, max_sentence_length]
    return tf.reduce_mean(loss_val * mask)


def compute_loss(l1, l2, logits, labels, lengths, trainable_vars):
    """ Computes loss with regularization """
    return compute_text_generation_loss(logits, labels, lengths) + compute_regularization_penalty(l1, l2, trainable_vars)
