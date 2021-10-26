""" Layer common utilities """
import pickle

import numpy as np
import tensorflow as tf
from absl import logging


def init_word_embedding(vocab_size, num_units, we_trainable, we_file=None, name_prefix="w"):
    """Initialize word embeddings from random initialization or pretrained word embedding """

    if not we_file:
        embedding_name = "{}_embedding".format(name_prefix)
        # Random initialization
        embedding = tf.compat.v1.get_variable(
            embedding_name, [vocab_size, num_units], dtype=tf.float32, trainable=we_trainable)
        logging.info(f'Initializing embedding {embedding_name}')
    else:
        # Initialize by pretrained word embedding
        embedding_name = "{}_pretrained_embedding".format(name_prefix)
        we = pickle.load(tf.io.gfile.GFile(we_file, 'rb'))
        assert vocab_size == we.shape[0] and num_units == we.shape[1]
        embedding = tf.compat.v1.get_variable(name=embedding_name,
                                              shape=[vocab_size, num_units],
                                              dtype=tf.float32,
                                              initializer=tf.compat.v1.constant_initializer(we),
                                              trainable=we_trainable)
        logging.info(f'Loading pretrained embedding {embedding_name} from {we_file}')
    return embedding


def get_sorted_dict(dct: dict):
    """Returns dictionary in sorted order"""
    return dict(sorted(dct.items()))


def inf(dtype):
    """Returns a value close to infinity, but is still finite in `dtype`.
    This is useful to get a very large value that is still zero when multiplied by zero. The floating-point "Inf" value is NaN when multiplied by zero.

    :param dtype: A dtype. The returned value will be finite when casted to this dtype.
    :return A very large value.
    """
    if dtype == "float32" or dtype == "bfloat16":
        return 1e7
    elif dtype == "float16":
        # Disable no-member lint error, as the linter thinks np.float16 does not
        # exist for some reason.
        return np.finfo(np.float16).max - 1  # pylint: disable=no-member
    else:
        raise AssertionError("Invalid dtype: %s" % dtype)


def expand_to_same_rank(tensor, target):
    """Expands a given tensor to target's rank to be broadcastable.

    :param tensor: input tensor to tile. Shape: [b, d1, ..., da]
    :param target: target tensor. Shape: [b, d1, ..., da, ..., dn]
    :return Tiled tensor of shape [b, d1, ..., da, 1, ..., 1] with same rank of target.
    :raise ValueError, if the shape rank of rank tensor/target is None.
    """
    if tensor.shape.rank is None:
        raise ValueError("Expect rank for tensor shape, but got None.")
    if target.shape.rank is None:
        raise ValueError("Expect rank for target shape, but got None.")

    with tf.name_scope("expand_rank"):
        diff_rank = target.shape.rank - tensor.shape.rank
        for _ in range(diff_rank):
            tensor = tf.expand_dims(tensor, -1)
        return tensor


def get_last_valid_elements(x, batch_size, seq_len):
    """Returns the last valid element in x

    :param x: input sequences. Shape=[batch_size, max_seq_len]
    :param batch_size: batch size
    :param seq_len: length of sequences for each sentence
    """
    indices = tf.stack([tf.range(batch_size), seq_len - 1], axis=1)
    last_elements = tf.gather_nd(x, indices)  # [batch_size, num_units]
    return last_elements


def _tile_batch(t, multiplier):
    """Core single-tensor implementation of tile_batch."""
    t = tf.convert_to_tensor(t, name="t")  # shape=[d0, d1, ..., dn]
    shape_t = tf.shape(t)
    if t.shape.ndims is None or t.shape.ndims < 1:
        raise ValueError("t must have statically known rank")
    tiling = [1] * (t.shape.ndims + 1)  # value=[1, 1, ..., 1] n+1 1s
    tiling[1] = multiplier  # value=[1, multiplier, 1, 1]
    tiled_static_batch_size = (
        t.shape[0] * multiplier if t.shape[0] is not None else None
    )
    tiled = tf.tile(tf.expand_dims(t, 1), tiling)  # shape=[d0, multiplier, d1, ..., dn]
    tiled = tf.reshape(tiled, tf.concat(([shape_t[0] * multiplier], shape_t[1:]), 0))  # shape=[d0*multiplier, d1, ..., dn]
    tiled.set_shape(tf.TensorShape([tiled_static_batch_size]).concatenate(t.shape[1:]))
    return tiled


def tile_batch(t, multiplier: int, name=None) -> tf.Tensor:
    """Tiles the batch dimension of a (possibly nested structure of) tensor(s).
    For each tensor t in a (possibly nested structure) of tensors, this function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed
    of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape `[batch_size * multiplier, s0, s1, ...]` composed of minibatch
    entries `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated `multiplier` times.

    :param t: `Tensor` shaped `[batch_size, ...]`.
    :param multiplier: Python int.
    :param name: Name scope for any created operations.

    :return A (possibly nested structure of) `Tensor` shaped `[batch_size * multiplier, ...]`.
    :raise ValueError: if tensor(s) `t` do not have a statically known rank or the rank is < 1.
    """
    with tf.name_scope(name or "tile_batch"):
        return tf.nest.map_structure(lambda t_: _tile_batch(t_, multiplier), t)


def _shape_list(tensor):
    """Return a list of the tensor's shape, and ensure no None values in list."""
    # Get statically known shape (may contain None's for unknown dimensions)
    shape = tensor.get_shape().as_list()

    # Ensure that the shape values are not None
    dynamic_shape = tf.shape(tensor)
    for i in range(len(shape)):  # pylint: disable=consider-using-enumerate
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape


def _get_shape_keep_last_dim(tensor):
    shape_list = _shape_list(tensor)

    # Only the last
    for i in range(len(shape_list) - 1):
        shape_list[i] = None

    if isinstance(shape_list[-1], tf.Tensor):
        shape_list[-1] = None
    return tf.TensorShape(shape_list)


def _log_prob_from_logits(logits):
    """ Returns log prob from logits
    :param logits: tf.Tensor(dtype=tf.float32) Shape=[batch_size, beam_width, vocab_size]
    """
    return logits - tf.reduce_logsumexp(logits, axis=2, keepdims=True)


def is_tf_function(func):
    """Returns whether the function is a tf.function decorated method"""
    return hasattr(func, 'get_concrete_function')


def get_tf_function_names(clz):
    """Returns the list of tf function names of the class"""
    methods = [method_name for method_name in dir(clz) if callable(getattr(clz, method_name))]
    return list(filter(
        lambda method_name: is_tf_function(getattr(clz, method_name)),
        methods))
