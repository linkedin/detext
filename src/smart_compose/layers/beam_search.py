"""Prefix aware beam search to find sequences with highest probabilities."""

import tensorflow as tf

from smart_compose.layers.prefix_search import PrefixSearcher
from smart_compose.utils.layer_utils import inf, expand_to_same_rank, _shape_list, _get_shape_keep_last_dim, _log_prob_from_logits, tile_batch, \
    get_last_valid_elements
from smart_compose.utils.parsing_utils import InternalFtrType, OutputFtrType


def _length_normalization(length_norm_power, length, dtype=tf.float32):
    """Returns length normalization factor."""
    return tf.pow(((5. + tf.cast(length, dtype)) / 6.), length_norm_power)


def expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
    :param tensor: tensor to tile [batch_size, ...]
    :param beam_size: How much to tile the tensor by.
    :return Tiled tensor [batch_size, beam_size, ...]
    """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def flatten_beam_dim(tensor):
    """Reshapes first two dimensions into a single dimension.
    :param tensor: Tensor to reshape of shape [A, B, ...]
    :return Reshaped tensor of shape [A*B, ...]
    """
    shape = _shape_list(tensor)
    shape[0] *= shape[1]
    shape.pop(1)  # Remove beam dim
    return tf.reshape(tensor, shape)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].
    :param tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
    :param batch_size: Tensor, original batch size.
    :param beam_size: int, original beam size.
    :return Reshaped tensor of shape [batch_size, beam_size, ...]
    """
    shape = _shape_list(tensor)
    new_shape = [batch_size, beam_size] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
    """Gathers beams from nested structure of tensors.
    Each tensor in nested represents a batch of beams, where beam refers to a single search state (beam search involves searching
    through multiple states in parallel).
    This function is used to gather the top beams, specified by beam_indices, from the nested tensors.

    :param nested: Nested structure (tensor, list, tuple or dict) containing tensors with shape [batch_size, beam_size, ...].
    :param beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each value in beam_indices must be between [0, beam_size), and are not
        necessarily unique.
    :param batch_size: int size of batch
    :param new_beam_size: int number of beams to be pulled from the nested tensors.
    :return Nested structure containing tensors with shape [batch_size, new_beam_size, ...]
    """
    # Compute the i'th coodinate that contains the batch index for gather_nd
    # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]
    batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

    # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor with shape [batch_size, beam_size, 2], where the
    #   last dimension contains the (i, j) gathering coordinates
    coordinates = tf.stack([batch_pos, beam_indices], axis=2)

    return tf.nest.map_structure(lambda state: tf.gather_nd(state, coordinates), nested)


class BeamSearchState:
    """Keys to dictionary storing the state of the beam search loop"""

    # Variable storing the loop index
    CUR_INDEX = "CUR_INDEX"

    # Top sequences that are alive for each batch item. Alive sequences are ones that have not generated an EOS token. Sequences that
    #   reach EOS are marked as finished and moved to the FINISHED_SEQ tensor
    ALIVE_SEQ = "ALIVE_SEQ"  # [batch_size, beam_size, CUR_INDEX + 1]
    # Log probabilities of each alive sequence. Shape [batch_size, beam_size]
    ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
    # Dictionary of cached values for each alive sequence. The cache stores the encoder output, attention bias,
    #   and the decoder attention output from the previous iteration.
    ALIVE_CACHE = "ALIVE_CACHE"

    # Top finished sequences for each batch item. Sequences that are shorter than CUR_INDEX + 1 are padded with 0s.
    FINISHED_SEQ = "FINISHED_SEQ"  # [batch_size, beam_size, CUR_INDEX + 1].
    # Scores for each finished sequence. Score = log probability / length norm
    FINISHED_SCORES = "FINISHED_SCORES"  # [batch_size, beam_size]
    # Flags indicating which sequences in the finished sequences are finished.
    FINISHED_FLAGS = "FINISHED_FLAGS"  # [batch_size, beam_size]


def sequence_beam_search(get_logits_and_cache_fn,
                         initial_ids,
                         initial_cache,
                         vocab_size,
                         beam_size,
                         length_norm_power,
                         max_decode_length,
                         sep_id,
                         pad_id,
                         min_seq_prob=0,
                         dtype="float32"):
    """Searches for sequence of subtoken ids with the largest probability.

    :param get_logits_and_cache_fn: A function that takes in ids, index, and cache as arguments. The passed in arguments will have shape:
                ids: Tensor(dtype=int, shape=[batch_size * beam_size, index])
                index: Tensor(dtype=int, shape=[])
                cache: A nested dictionary of tensors [batch_size * beam_size, ...].
            The function must return a tuple of logits and new cache:
                logits: Tensor(dtype=float32, shape=[batch * beam_size, vocab_size])
                new cache: A nested dictionary with the same shape/structure as the inputted cache.
    :param initial_ids: Tensor(dtype=int, shape=[batch_size]). Starting ids for each batch item.
    :param initial_cache: A dictionary, containing starting decoder variables information.
    :param vocab_size: int.
    :param beam_size: int.
    :param length_norm_power: float. Strength of length normalization.
    :param max_decode_length: int. Maximum length of the decoded sequence
    :param sep_id: int. ID of eos token, used to determine when a sequence has finished
    :param dtype: The data type used for score computation. The default is tf.float32
    :return: Top decoded sequences [batch_size, beam_size, max_decode_length]
            sequence scores [batch_size, beam_size]
    """
    searcher = BeamSearcher(get_logits_and_cache_fn, vocab_size, beam_size, length_norm_power,
                            max_decode_length, sep_id, pad_id, min_seq_prob, dtype)
    return searcher.search(initial_ids, initial_cache)


class BeamSearcher(tf.Module):
    def __init__(self,
                 get_cache_and_logits_fn,
                 vocab_size,
                 beam_size,
                 length_norm_power,
                 max_decode_length,
                 eos_id,
                 pad_id,
                 min_seq_prob,
                 dtype=tf.float32):
        """Initializes beam search
        :param get_cache_and_logits_fn: A function to provide logits. The passed in arguments are:
                    ids: tf.Tensor(dtype=tf.int, shape=[batch_size * beam_size, index])
                    index: tf.Tensor(dtype=tf.int, shape=[])
                    cache: Nested dictionary of tf.Tensor(shape=[batch_size * beam_size, ...]).
                The function must return a tuple of logits and the updated cache. More specifically,
                    logits: tf.Tensor(shape=[batch * beam_size, vocab_size])
                    updated cache: Nested dictionary with the same structure as the input cache
        :param vocab_size: int. Size of the vocabulary
        :param beam_size: int. Number of beams
        :param length_norm_power: float. Strength of length normalization
        :param max_decode_length: int. Maximum number of steps to decode a sequence
        :param eos_id: int. ID of end of sentence token
        :param min_seq_prob: float. Minimum probability of the emitted sequence. If set to zero, then no pruning will be performed
        :param dtype: Data type used for score computation
        """
        self.get_cache_and_logits_fn = get_cache_and_logits_fn
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.length_norm_power = length_norm_power
        self.max_decode_length = max_decode_length
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.dtype = tf.as_dtype(dtype)
        self.enable_pruning = min_seq_prob > 0
        self.min_log_seq_prob = -inf(self.dtype) if min_seq_prob <= 0 else tf.math.log(min_seq_prob)

    def search(self, initial_ids, initial_cache):
        """Beam search for sequences with highest scores
        :param initial_ids: tf.Tensor(dtype=int, shape=[batch_size, 1]. Initial ids to pass into the symbols_to_logits_fn.
        :param initial_cache: dictionary storing values to be passed into the symbols_to_logits_fn
        :return finished_seq and finished_scores
        """
        batch_size = tf.shape(initial_ids)[0]
        state, state_shapes = self._create_initial_state(initial_ids, initial_cache, batch_size)

        def _grow_alive_seq(state):
            """Grows alive sequences by one token, collect top 2*beam_size sequences.
            :param state: A dictionary with the current loop state.
            :return Tuple of (
                        Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1],
                        Scores of returned sequences [batch_size, 2 * beam_size],
                        New alive cache, for each of the 2 * beam_size sequences
                    )
            """
            i = state[BeamSearchState.CUR_INDEX]  # []
            alive_seq = state[BeamSearchState.ALIVE_SEQ]  # [batch_size, beam_width, seq_len]
            alive_log_probs = state[BeamSearchState.ALIVE_LOG_PROBS]  # [batch_size, beam_width]
            alive_cache = state[BeamSearchState.ALIVE_CACHE]  # Dictionary with tensors shape [batch_size, beam_width, ...]

            beams_to_keep = 2 * self.beam_size

            # Get logits for the next candidate IDs for the alive sequences. Get the new cache values at the same time.
            flat_ids = flatten_beam_dim(alive_seq)  # [batch_size * beam_size]
            flat_cache = tf.nest.map_structure(flatten_beam_dim, alive_cache)

            flat_logits, flat_cache = self.get_cache_and_logits_fn(flat_ids, i, flat_cache)

            # Unflatten logits to shape [batch_size, beam_size, vocab_size]
            logits = _unflatten_beam_dim(flat_logits, batch_size, self.beam_size)
            new_cache = tf.nest.map_structure(lambda t: _unflatten_beam_dim(t, batch_size, self.beam_size), flat_cache)

            # Convert logits to normalized log probs
            candidate_log_probs = _log_prob_from_logits(logits)

            # Calculate new log probabilities if each of the alive sequences were extended # by the the candidate IDs.
            # Shape [batch_size, beam_size, vocab_size]
            log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

            # Pruning: set the probabilities of sequences that have prob < min_prob to be -inf
            if self.enable_pruning:
                # Set log prob of probs lower than threshold to be -inf
                non_confident_continuation = tf.less_equal(log_probs, self.min_log_seq_prob)  # [batch_size, beam_size, vocab_size]
                log_probs = log_probs + tf.cast(non_confident_continuation, dtype=tf.float32) * (-inf(self.dtype))  # [batch_size, beam_size, vocab_size]

                # Set log prob of [PAD] of beams that has no qualified next tokens (tokens that result in seq prob >= threshold) to be 0. Therefore, the
                #   log prob of alive seq + [PAD] is the same as the log prob of alive seq
                all_non_confident_continuation = tf.math.reduce_all(non_confident_continuation, axis=2, keepdims=True)  # [batch_size, beam_size, 1]

                alive_log_probs_expanded = tf.expand_dims(alive_log_probs, axis=2) + tf.zeros_like(log_probs)  # [batch_size, beam_size, vocab_size]
                mask = tf.concat([
                    tf.zeros(shape=[batch_size, self.beam_size, self.pad_id], dtype=tf.bool),
                    tf.cast(all_non_confident_continuation, dtype=tf.bool),
                    tf.zeros(shape=[batch_size, self.beam_size, self.vocab_size - self.pad_id - 1], dtype=tf.bool)
                ], axis=2)  # [batch_size, beam_size, vocab_size]

                log_probs = tf.where(mask, alive_log_probs_expanded, log_probs)

            # Each batch item has beam_size * vocab_size candidate sequences. For each batch item, get the k candidates with the highest log probabilities.
            flat_log_probs = tf.reshape(log_probs, [-1, self.beam_size * self.vocab_size])
            topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)

            # Calculate the beam indices of alive seq that generates the highest next seq log probabilities
            topk_beam_indices = topk_indices // self.vocab_size  # [batch_size, beams_to_keep]

            # Extract the alive sequences that generate the highest log probabilities after being extended
            topk_seq, new_cache = _gather_beams([alive_seq, new_cache],
                                                topk_beam_indices, batch_size,
                                                beams_to_keep)

            # Append the most probable IDs to the topk sequences
            topk_ids = topk_indices % self.vocab_size
            topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)
            return topk_seq, topk_log_probs, topk_ids, new_cache

        def _get_new_alive_state(new_seq, new_log_probs, new_finished_flags, new_cache):
            """Gathers the top k sequences that are still alive.
            :param new_seq: tf.Tensor(dtype=int32, shape=[batch_size, 2 * beam_size, cur_index + 1]).
                New sequences generated by growing the current alive sequences
            :param new_log_probs: tf.Tensor(dtype=float32, shape=[batch_size, beam_size]). Log probabilities of new sequences
            :param new_finished_flags: tf.Tensor(dtype=bool, shape=[batch_size, beam_size]). Indicator of which sequences are live inside the beam.
            :param new_cache: Dict of cached values for each sequence.
            :return Dictionary with alive keys from _StateKeys:
                {Top beam_size sequences that are still alive (don't end with eos_id)
                 Log probabilities of top alive sequences
                 Dict cache storing decoder states for top alive sequences}
            """
            # To prevent finished sequences from being considered, set log probs to -inf
            new_log_probs += tf.cast(new_finished_flags, self.dtype) * -inf(self.dtype)

            _, topk_indexes = tf.nn.top_k(new_log_probs, k=self.beam_size)
            top_alive_seq, top_alive_log_probs, top_alive_cache = (
                _gather_beams([new_seq, new_log_probs, new_cache], topk_indexes, batch_size, self.beam_size))

            return {BeamSearchState.ALIVE_SEQ: top_alive_seq,
                    BeamSearchState.ALIVE_LOG_PROBS: top_alive_log_probs,
                    BeamSearchState.ALIVE_CACHE: top_alive_cache}

        def _get_new_finished_state(state, new_seq, new_log_probs, new_finished_flags):
            """Combine new and old finished sequences, and gather the top k sequences.
            :param state: A dictionary with the current loop state.
            :param new_seq: New sequences generated by growing the current alive sequences
                int32 tensor with shape [batch_size, beam_size, i + 1]
            :param new_log_probs: Log probabilities of new sequences float32 tensor with
                shape [batch_size, beam_size]
            :param new_finished_flags: A boolean Tensor indicates which sequences are live
                inside the beam.
            :return Dictionary with finished keys from _StateKeys:
                {
                    Top beam_size finished sequences based on score,
                     Scores of finished sequences,
                     Finished flags of finished sequences
                 }
            """
            i = state[BeamSearchState.CUR_INDEX]
            finished_seq = state[BeamSearchState.FINISHED_SEQ]
            finished_scores = state[BeamSearchState.FINISHED_SCORES]
            finished_flags = state[BeamSearchState.FINISHED_FLAGS]

            # First append a column of 0-ids to finished_seq to increment the length.
            # New shape of finished_seq: [batch_size, beam_size, i + 1]
            finished_seq = tf.concat([finished_seq, tf.ones([batch_size, self.beam_size, 1], tf.int32) * self.pad_id], axis=2)

            # Calculate new seq scores from log probabilities.
            length_norm = _length_normalization(self.length_norm_power, i + 1, dtype=self.dtype)
            new_scores = new_log_probs / length_norm

            # Set the scores of the still-alive seq in new_seq to large negative values.
            new_scores += ((1. - tf.cast(new_finished_flags, self.dtype)) * -inf(self.dtype))

            # Combine sequences, scores, and flags.
            finished_seq = tf.concat([finished_seq, new_seq], axis=1)
            finished_scores = tf.concat([finished_scores, new_scores], axis=1)
            finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

            # Return the finished sequences with the best scores.
            # Rerank the finished sequences because scores of the newly finished sequences could be higher than those of the existing finished ones
            _, topk_indexes = tf.nn.top_k(finished_scores, k=self.beam_size)
            top_finished_seq, top_finished_scores, top_finished_flags = (
                _gather_beams([finished_seq, finished_scores, finished_flags],
                              topk_indexes, batch_size, self.beam_size))

            return {
                BeamSearchState.FINISHED_SEQ: top_finished_seq,
                BeamSearchState.FINISHED_SCORES: top_finished_scores,
                BeamSearchState.FINISHED_FLAGS: top_finished_flags
            }

        def _search_step(state):
            """Beam search loop body
            Grow alive sequences by a single ID. Sequences that have reached the EOS token are marked as finished. The alive and finished
                sequences with the highest log probabilities and scores are returned

            :param state: A dictionary with the current loop state
            :return new state dictionary
            """
            # Grow alive sequences by one token
            new_seq, new_log_probs, topk_ids, new_cache = _grow_alive_seq(state)
            new_finished_flags = tf.logical_or(tf.equal(topk_ids, self.eos_id), tf.equal(topk_ids, self.pad_id))

            # Collect top beam_size alive sequences
            alive_state = _get_new_alive_state(new_seq, new_log_probs, new_finished_flags, new_cache)

            # Combine newly finished sequences with existing finished sequences, and collect the top k scoring sequences.
            finished_state = _get_new_finished_state(state, new_seq, new_log_probs, new_finished_flags)

            # Increment loop index and create new state dictionary
            new_state = {BeamSearchState.CUR_INDEX: state[BeamSearchState.CUR_INDEX] + 1}
            new_state.update(alive_state)
            new_state.update(finished_state)
            return [new_state]

        finished_state = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(
                self._continue_search,
                _search_step,
                loop_vars=[state],
                shape_invariants=[state_shapes],
                parallel_iterations=1))
        finished_state = finished_state[0]
        return self._process_finished_state(finished_state)

    def _process_finished_state(self, finished_state):
        alive_seq = finished_state[BeamSearchState.ALIVE_SEQ]
        alive_log_probs = finished_state[BeamSearchState.ALIVE_LOG_PROBS]
        finished_seq = finished_state[BeamSearchState.FINISHED_SEQ]
        finished_scores = finished_state[BeamSearchState.FINISHED_SCORES]
        finished_flags = finished_state[BeamSearchState.FINISHED_FLAGS]
        finished_cond = tf.reduce_any(finished_flags, 1, name="finished_cond")
        seq_cond = expand_to_same_rank(finished_cond, finished_seq)
        score_cond = expand_to_same_rank(finished_cond, finished_scores)

        # Account for corner case where there are no finished sequences for a particular batch item. In that case, return alive sequences for that batch item
        finished_seq = tf.where(seq_cond, finished_seq, alive_seq)
        finished_scores = tf.where(score_cond, finished_scores, alive_log_probs)
        return finished_seq, finished_scores

    def _create_initial_state(self, initial_ids, initial_cache, batch_size):
        """Return initial state dictionary and its shape invariants."""
        for key, value in initial_cache.items():
            for inner_value in tf.nest.flatten(value):
                if inner_value.dtype != self.dtype:
                    raise TypeError(
                        "initial_cache element for key '%s' has dtype %s that does not "
                        "match BeamSearch's dtype of %s. Value: %s" %
                        (key, inner_value.dtype.name, self.dtype.name, inner_value))

        # Current loop index (starts at 0)
        cur_index = tf.constant(0)

        # Create alive sequence with shape [batch_size, beam_size, 1]
        alive_seq = expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = tf.expand_dims(alive_seq, axis=2)

        # Create tensor for storing initial log probabilities.
        # Assume initial_ids are prob 1.0
        initial_log_probs = tf.constant([[0.] + [-float("inf")] * (self.beam_size - 1)], dtype=self.dtype)
        alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])

        # Expand all values stored in the dictionary to the beam size, so that each beam has a separate cache.
        alive_cache = tf.nest.map_structure(lambda t: expand_to_beam_size(t, self.beam_size), initial_cache)

        # Initialize tensor storing finished sequences with filler values.
        finished_seq = tf.ones(tf.shape(alive_seq), tf.int32) * self.pad_id

        # Set scores of the initial finished seqs to negative infinity.
        finished_scores = tf.ones([batch_size, self.beam_size],
                                  dtype=self.dtype) * -inf(self.dtype)

        # Initialize finished flags with all False values.
        finished_flags = tf.zeros([batch_size, self.beam_size], tf.bool)

        # Create state dictionary
        state = {
            BeamSearchState.CUR_INDEX: cur_index,
            BeamSearchState.ALIVE_SEQ: alive_seq,
            BeamSearchState.ALIVE_LOG_PROBS: alive_log_probs,
            BeamSearchState.ALIVE_CACHE: alive_cache,
            BeamSearchState.FINISHED_SEQ: finished_seq,
            BeamSearchState.FINISHED_SCORES: finished_scores,
            BeamSearchState.FINISHED_FLAGS: finished_flags
        }

        state_shape_invariants = {
            BeamSearchState.CUR_INDEX:
                tf.TensorShape([]),
            BeamSearchState.ALIVE_SEQ:
                tf.TensorShape([None, self.beam_size, None]),
            BeamSearchState.ALIVE_LOG_PROBS:
                tf.TensorShape([None, self.beam_size]),
            BeamSearchState.ALIVE_CACHE:
                tf.nest.map_structure(_get_shape_keep_last_dim, alive_cache),
            BeamSearchState.FINISHED_SEQ:
                tf.TensorShape([None, self.beam_size, None]),
            BeamSearchState.FINISHED_SCORES:
                tf.TensorShape([None, self.beam_size]),
            BeamSearchState.FINISHED_FLAGS:
                tf.TensorShape([None, self.beam_size])
        }

        return state, state_shape_invariants

    def _continue_search(self, state):
        """Returns whether to continue the search loop.
        The loops should terminate when
          1) when decode length has been reached, or
          2) when the worst score in the finished sequences is better than the best score in the alive sequences (i.e.
            the finished sequences are provably unchanging)
        :param state: A dictionary with the current loop state.
        :return Bool tensor with value True if loop should continue, False if loop should terminate.
        """
        i = state[BeamSearchState.CUR_INDEX]
        alive_log_probs = state[BeamSearchState.ALIVE_LOG_PROBS]
        finished_scores = state[BeamSearchState.FINISHED_SCORES]
        finished_flags = state[BeamSearchState.FINISHED_FLAGS]

        not_at_max_decode_length = tf.less(i, self.max_decode_length)

        # Calculate largest length penalty (the larger penalty, the better score).
        max_length_norm = _length_normalization(self.length_norm_power, self.max_decode_length, dtype=self.dtype)
        # Get the best possible scores from alive sequences.
        # This tf.slice/tf.squeeze is equivalent to alive_log_probs[:, 0] which emits a tf.strided_slice. tf.slice is easier to reason about as we aren't
        #   actually taking a non trivial stride.
        best_alive_scores = tf.squeeze(tf.slice(alive_log_probs, [0, 0], [-1, 1]), axis=1) / max_length_norm

        # Compute worst score in finished sequences for each batch element
        finished_scores *= tf.cast(finished_flags, self.dtype)  # set filler scores to zero
        lowest_finished_scores = tf.reduce_min(finished_scores, axis=1)

        # If there are no finished sequences in a batch element, then set the lowest finished score to -INF for that element.
        finished_batches = tf.reduce_any(finished_flags, 1)
        lowest_finished_scores += ((1.0 - tf.cast(finished_batches, self.dtype)) * -inf(self.dtype))

        # If the worst score of finished sequence is better than that of the best alive score. Then stop searching
        worst_finished_score_better_than_best_alive_score = tf.reduce_all(tf.greater(lowest_finished_scores, best_alive_scores))

        return tf.logical_and(not_at_max_decode_length, tf.logical_not(worst_finished_score_better_than_best_alive_score))


class PrefixAwareBeamSearcher(tf.Module):
    """Prefix aware beam searcher"""

    def __init__(self,
                 vocab_size,
                 beam_width,
                 length_norm_power,
                 max_decode_length,
                 sep_id,
                 pad_id,
                 min_seq_prob,
                 vocab_layer,
                 inference_min_len,
                 inference_max_len,
                 num_cls_inference,
                 num_sep_inference
                 ):
        """ Initializes prefix aware beam search

        :param vocab_size: int. Vocabulary size. The number of the tokens
        :param beam_width: int. Beam width
        :param length_norm_power: float. Normalization power to penalize long decoding sequences. The larger the more penalty
        :param max_decode_length: int. Maximum decoding length
        :param sep_id: int. ID of segment (e.g. sentence) separator
        :param pad_id: int. ID of the padding token
        :param min_seq_prob: float. Minimum probability of the emitted sequence. If set to zero, then no pruning will be performed
        :param vocab_layer: Vocabulary layer.
        :param inference_min_len: int. Min inference length.
        :param inference_max_len: int. Max inference length. Input sentences larger than the length will be trimmed to this number (take the last
            inference_max_len tokens).
        :param num_cls_inference: Number of CLS tokens to add to the start of the sentence. For text encoders like CNN, this needs to be set according
            to the filter window size
        :param num_sep_inference: Number of SEP tokens to add to the end of the sentence. For text encoders like CNN, this needs to be set according
            to the filter window size
        """
        self._vocab_size = vocab_size
        self._beam_width = beam_width
        self._length_norm_power = length_norm_power
        self._max_decode_length = max_decode_length

        self._sep_id = sep_id
        self._pad_id = pad_id

        self._min_seq_prob = min_seq_prob

        self._num_cls_inference = num_cls_inference
        self._num_sep_inference = num_sep_inference

        self._vocab_layer = vocab_layer
        self._inference_min_len = inference_min_len
        self._inference_max_len = inference_max_len

        self.prefix_searcher = PrefixSearcher(self._vocab_layer, self._inference_min_len, self._inference_max_len,
                                              self._num_cls_inference, self._num_sep_inference)

    def convert_ids_to_texts(self, predicted_ids):
        """Converts IDs to text strings"""
        batch_size = tf.shape(predicted_ids)[0]
        beam_width = tf.shape(predicted_ids)[1]
        predicted_ids = tf.reshape(predicted_ids, [batch_size * beam_width, -1])  # [batch_size*beam_width, max_decode_length]
        predicted_texts = self._vocab_layer.convert_ids_to_texts(predicted_ids)  # [batch_size*beam_width]
        return tf.reshape(predicted_texts, [batch_size, beam_width])

    def __call__(self, inputs, get_logits_and_cache_fn, get_initial_logits_and_cache_fn):
        """ Performs prefix aware beam search

        :param inputs: Tensor(dtype=string, shape=[]). Input text sequence
        :param get_logits_and_cache_fn: A function that takes in ids, index, and cache as arguments. The passed in arguments will have shape:
                ids: Tensor(dtype=int, shape=[batch_size * beam_size, index])
                index: Tensor(dtype=int, shape=[])
                cache: A nested dictionary of tensors [batch_size * beam_size, ...].
            The function must return a tuple of logits and new cache:
                logits: Tensor(dtype=float32, shape=[batch * beam_size, vocab_size])
                new cache: A nested dictionary with the same shape/structure as the inputted cache.
        :param get_initial_logits_and_cache_fn: a function that takes in tokenized ids and return the initial_logits. The passed in arguments will have shape:
                tokenized_ids: Tensor(dtype=int, shape=[])
            The function must return a tuple of logits and new cache:
                initial_logits: Tensor(dtype=float32, shape=[batch * beam_size, vocab_size])
                initial_cache: A nested dictionary with the same shape/structure as the inputted cache.
        :return: A dictionary containing
            {
                EXIST_PREFIX: Tensor(dtype=bool). Whether there are tokens starting with the given prefix
                PREDICTED_SCORES: Tensor(dtype=float, shape=[batch_size(1), beam_size]). Scores of the predicted completion
                PREDICTED_TEXTS: Tensor(dtype=string, shape=[batch_size(1), beam_size]). Predicted texts
            }
        """
        squeezed_inputs = tf.squeeze(inputs, axis=0)
        prefix_search_results = self.prefix_searcher(squeezed_inputs)

        def get_empty_results():
            predicted_ids = tf.zeros([tf.shape(inputs)[0], self._beam_width, self._max_decode_length + 1], dtype=tf.int32)
            return {
                OutputFtrType.EXIST_PREFIX: tf.zeros([tf.shape(inputs)[0]], dtype=tf.bool),
                OutputFtrType.PREDICTED_SCORES: tf.zeros([tf.shape(inputs)[0], self._beam_width], dtype=tf.float32),
                OutputFtrType.PREDICTED_TEXTS: self.convert_ids_to_texts(predicted_ids)
            }

        def get_prefixed_results():
            tokenized_ids = tf.expand_dims(prefix_search_results[InternalFtrType.SEQUENCE_TO_ENCODE], axis=0)
            length = tf.expand_dims(prefix_search_results[InternalFtrType.LENGTH], axis=0)

            start_tokens = get_last_valid_elements(tokenized_ids, tf.shape(inputs)[0], length)
            offset = tf.cast(tf.logical_not(prefix_search_results[InternalFtrType.COMPLETION_VOCAB_MASK]), dtype=tf.float32) * -inf(tf.float32)
            offset = tf.expand_dims(offset, axis=0)
            tiled_offset = tile_batch(offset, self._beam_width)

            initial_logits, initial_cache = get_initial_logits_and_cache_fn(tokenized_ids)
            masked_logits = initial_logits + tiled_offset

            def prefix_aware_get_logits_and_cache_fn(ids, i, cache):
                return tf.cond(i == 0, true_fn=lambda: (masked_logits, initial_cache), false_fn=lambda: get_logits_and_cache_fn(ids, i, cache))

            predicted_ids, scores = sequence_beam_search(
                get_logits_and_cache_fn=prefix_aware_get_logits_and_cache_fn,
                initial_ids=start_tokens,
                initial_cache=initial_cache,
                vocab_size=self._vocab_size,
                beam_size=self._beam_width,
                length_norm_power=self._length_norm_power,
                max_decode_length=self._max_decode_length,
                sep_id=self._sep_id,
                pad_id=self._pad_id,
                min_seq_prob=self._min_seq_prob,
                dtype=tf.float32)

            return {
                OutputFtrType.EXIST_PREFIX: tf.ones([tf.shape(inputs)[0]], dtype=tf.bool),
                OutputFtrType.PREDICTED_SCORES: scores,
                OutputFtrType.PREDICTED_TEXTS: self.convert_ids_to_texts(predicted_ids)
            }

        return tf.cond(prefix_search_results[OutputFtrType.EXIST_PREFIX], true_fn=get_prefixed_results, false_fn=get_empty_results)
