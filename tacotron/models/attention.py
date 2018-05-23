import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention

from hparams import hparams


# From https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
def compute_attention(attention_mechanism, cell_output, attention_state,
                      attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
    This attention is described in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
        gio, “Attention-based models for speech recognition,” in Ad-
        vances in Neural Information Processing Systems, 2015, pp.
        577–585.
    #############################################################################
            hybrid attention (content-based + location-based)
                        f = F * α_{i-1}
            energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
    #############################################################################
    Args:
        W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
        W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
        W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
    Returns:
        A '[batch_size, max_time]' attention score (energy)
    """
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or tf.shape(W_keys)[-1]

    v_a = tf.get_variable(
        'attention_variable', shape=[num_units], dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable(
        'attention_bias', shape=[num_units], dtype=dtype,
        initializer=tf.zeros_initializer())

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax
    Introduced in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
        gio, “Attention-based models for speech recognition,” in Ad-
        vances in Neural Information Processing Systems, 2015, pp.
        577–585.
    ############################################################################
                        Smoothing normalization function
                a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
    ############################################################################
    Args:
        e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
            values of an attention mechanism
    Returns:
        matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
            attendance to multiple memory time steps.
    """
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class LocationSensitiveAttention(BahdanauAttention):
    """
    LocationSensitiveAttention which is described in https://arxiv.org/pdf/1508.01991.pdf
    """

    def __init__(self, num_units, memory, mask_encoder=True, memory_sequence_length=None, smoothing=False,
                 cumulate_weights=True,
                 name='LocationSensitiveAttention'):
        """Construct the Attention mechanism.

        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                    tensor should be shaped `[batch_size, max_time, ...]`.
            mask_encoder:  Boolean, whether to mask encoder paddings.
            memory_sequence_length: Sequence lengths for the batch entries
                    in memory.  If provided, the memory tensor rows are masked with zeros
                    for values past the respective sequence lengths.
            smoothing: whether use sigmoid as the score function default is softmax
            cumulate_weights: whether add the previous alignment
            name: the attention name
        """
        # select the score function
        normalization_function = _smoothing_normalization if (smoothing == True) else None
        memory_length = memory_sequence_length if (mask_encoder == True) else None

        super(LocationSensitiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_length,
            probability_fn=normalization_function,
            name=name
        )
        # make it take into account the alignment produced at the previous step
        self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
                                                     kernel_size=hparams.attention_kernel, padding='same',
                                                     use_bias=False,
                                                     name='location_features_convolution')
        self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
                                              dtype=tf.float32, name='location_features_layer')
        self._cumulate = cumulate_weights

    def __call__(self, query, state):
        """Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        previous_alignments = state
        with tf.variable_scope(None, "Location_Sensitive_Attention", [query]):
            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query
            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)

            # processed_location_features shape [batch_size, max_time, attention dimension]
            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)
            # Projected location features [batch_size, max_time, attention_dim]
            processed_location_features = self.location_layer(f)

            # energy shape [batch_size, max_time]
            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

            # alignments shape = energy shape = [batch_size, max_time]
            alignments = self._probability_fn(energy, previous_alignments)

            # Cumulate alignments
            if self._cumulate:
                next_state = alignments + previous_alignments
            else:
                next_state = alignments

            return alignments, next_state
