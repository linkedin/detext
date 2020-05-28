import pickle
import tensorflow as tf


def init_word_embedding(hparams, mode, name_prefix="w"):
    """Initialize word embeddings from random initialization or pretrained word embedding.

    This function is only used by encoding models other than BERT
    """

    we_trainable = hparams.get("we_trainable")
    we_file = hparams.get("we_file")
    vocab_size = hparams.get("vocab_size")
    num_units = hparams.get("num_units")

    if we_file is None:
        embedding_name = "{}_pretrained_embedding".format(name_prefix)
        # Random initialization
        embedding = tf.get_variable(
            embedding_name, [vocab_size, num_units], dtype=tf.float32, trainable=we_trainable)
    else:
        # Initialize by pretrained word embedding
        embedding_name = "{}_embedding".format(name_prefix)
        we = pickle.load(tf.gfile.Open(we_file, 'rb'))
        assert vocab_size == we.shape[0] and num_units == we.shape[1]
        embedding = tf.get_variable(
            name=embedding_name,
            shape=[vocab_size, num_units],
            dtype=tf.float32,
            initializer=tf.constant_initializer(we),
            trainable=we_trainable
        )
    return embedding


def _single_cell(unit_type, num_units, forget_bias, dropout,
                 mode, residual_connection=False, device_str=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    # Cell Type
    if unit_type == "lstm":
        single_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.nn.rnn_cell.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)

    # Device Wrapper
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)

    return single_cell


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode):
    """Create multi-layer RNN cell.

    Args:
      unit_type: string representing the unit type, i.e. "lstm".
      num_units: the depth of each unit.
      num_layers: number of cells.
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
        cells in the returned list will be wrapped with `ResidualWrapper`.
      forget_bias: the initial forget bias of the RNNCell(s).
      dropout: floating point value between 0.0 and 1.0:
        the probability of dropout.  this is ignored if `mode != TRAIN`.
      mode: either tf.contrib.learn.TRAIN/EVAL
      num_gpus: The number of gpus to use when performing round-robin
        placement of layers.
      base_gpu: The gpu device id to use for the first RNN cell in the
        returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
        as its device id.
      single_cell_fn: allow for adding customized cell.
        When not specified, we default to _single_cell
    Returns:
      An `RNNCell` instance.
    """
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers),
        )
        cell_list.append(single_cell)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)
