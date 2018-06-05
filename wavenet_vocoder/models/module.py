import tensorflow as tf

from wavenet_vocoder.util import sequence_mask
from wavenet_vocoder.models.mixture import discretized_mix_logistic_loss


class Embedding(object):
    """Embedding class for global conditions."""

    def __init__(self, num_embeddings, embedding_dim, std=0.1, name='gc_embedding'):
        # create embedding table
        self.embedding_table = tf.get_variable(name, [num_embeddings, embedding_dim],
                                               initializer=tf.truncated_normal_initializer(mean=0, std=std))

    def __call__(self, inputs):
        return tf.nn.embedding_lookup(self.embedding_table, inputs)


class ReLU(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, inputs):
        return tf.nn.relu(inputs, name=self.name)


class Conv1d1x1(tf.layers.Conv1D):
    def __init__(self, filters,
                 kernel_size=1,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 padding=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv1d1x1, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )
        # for incremental forward
        self.clear_buffer()
        self.padding_len = padding
        self._linearized_weight = None

    def set_mode(self, is_training):
        self.is_training = is_training

    def call(self, inputs):
        if self.padding_len is not None:
            if self.data_format == 'channels_first':
                inputs = tf.pad(inputs, tf.constant([[0, 0], [0, 0], [self.padding_len, 0]], dtype=tf.int32))
            else:
                inputs = tf.pad(inputs, tf.constant([(0, 0,), (self.padding_len, 0), (0, 0)]))
        return super(Conv1d1x1, self).call(inputs)

    def incremental_forward(self, inputs):
        """At sequential inference times:
            we adopt fast wavenet convolution queues by saving precomputed states for faster generation
            inputs: [batch_size, time_length, channels] ('NWC')! Channels last!
        """
        if self.is_training:
            raise RuntimeError('incremental_step only supports eval mode')

        # reshape weight
        weight = self._get_linearized_weight(inputs)
        kw = self.kernel_size
        dilation = self.dilation_rate

        batch_size = tf.shape(inputs)[0]

        if kw > 1:
            if self.input_buffer is None:
                self.input_buffer = tf.zeros((batch_size, kw+(kw-1)*(dilation-1), tf.shape(inputs)[2]))
            else:
                self.input_buffer = self.input_buffer[:, 1:, :]

            # append next input
            self.input_buffer = tf.concat([self.input_buffer, tf.expand_dims(inputs[:, -1, :], 1)], axis=1)

            inputs = self.input_buffer
            if dilation > 1:
                inputs = inputs[:, 0::dilation, :]

            # compute step prediction
            output = tf.matmul(tf.reshape(inputs, [batch_size, -1]), weight)
            if self.use_bias:
                output = tf.nn.bias_add(output, self.bias)
            return tf.reshape(output, [batch_size, 1, self.filters])

    def _get_linearized_weight(self, inputs):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        input_channel = tf.shape(inputs)[channel_axis]

        if self._linearized_weight is None:
            kw = self.kernel_size

            if tf.shape(self.kernel) == (self.filters, input_channel, kw):
                weight = tf.transpose(self.kernel, [2, 1, 0])
            else:
                # layers.Conv1D
                weight = self.kernel

            assert weight.shape == (kw, input_channel, self.filters)
            self._linearized_weight = tf.cast(tf.reshape(weight, [-1, self.filters]), dtype=inputs.dtype)
        return self._linearized_weight

    def clear_buffer(self):
        self.input_buffer = None


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 Step"""
    if is_incremental:
        return conv.incremental_forward(x)
    else:
        return conv(x)


class ResidualConv1dGLU(object):
    """Residual dilated conv1d + Gated Linear Unit."""
    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels=None, cin_channels=-1,
                 gin_channels=-1, dropout=1 - .95, padding=None, dilation=1, causal=True, use_bias=True,
                 name='ResidualConv1dGLU'):
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels

        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation

        self.causal = causal

        self.conv = Conv1d1x1(gate_channels, kernel_size, dilation_rate=dilation, padding=padding, use_bias=use_bias,
                                 name='residual_block_conv')

        # local condition
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(gate_channels, use_bias=use_bias, name='residual_block_cin_conv')
        else:
            self.conv1x1c = None

        # global condition
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gate_channels,  use_bias=use_bias, name='residual_block_gin_conv')
        else:
            self.conv1x1g = None

        self.conv1x1_out = Conv1d1x1(residual_channels, use_bias=use_bias, name='residual_block_out_conv')
        self.conv1x1_skip = Conv1d1x1(skip_out_channels,  use_bias=use_bias, name='residual_block_skip_conv')

    def set_mode(self, is_training):
        for conv in [self.conv, self.conv1x1c, self.conv1x1g, self.conv1x1_out, self.conv1x1_skip]:
            try:
                conv.set_mode(is_training)
            except AttributeError:
                pass

    def __call__(self, x, c=None, g=None):
        return self.forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self.forward(x, c, g, True)

    def forward(self, x, c, g, is_incremental):
        """
        Args:
            x: Tensor [batch_size, channels, time_length]
            c: Tensor [batch_size, c_channels, time_length]. Local conditioning features
            g: Tensor [batch_size, g_channels, time_length], global conditioning features
            is_incremental: Boolean, whether incremental mode is on
        Returns:
            Tensor output
        """
        residual = x
        x = tf.layers.dropout(x, rate=self.dropout, training=not is_incremental)
        if is_incremental:
            split_dim = -1
            x = self.conv.incremental_forward(x)
        else:
            split_dim = -1
            x = self.conv(x)
            x = x[:, :, :tf.shape(residual)[-1]] if self.causal else x

        a, b = tf.split(x, num_or_size_splits=2, axis=split_dim)

        # local condition
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = tf.split(c, num_or_size_splits=2, axis=split_dim)
            a, b = a + ca, b + cb

        # global condition
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = tf.split(g, num_or_size_splits=2, axis=split_dim)
            a, b = a + ga, b + gb

        x = tf.nn.tanh(a) * tf.nn.sigmoid(b)

        # skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)
        # residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = (x + residual) * tf.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for conv in [self.conv, self.conv1x1_out, self.conv1x1_skip,
                     self.conv1x1c, self.conv1x1g]:
            if conv is not None:
                conv.clear_buffer()


class ConvTransposed2d(object):
    """Use transposed conv to upsample"""
    def __init__(self, filters, kernel_size, freq_axis_kernel_size, padding, strides):
        self.convt = tf.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                               kernel_initializer=tf.constant_initializer(1 / freq_axis_kernel_size,
                                                                                          dtype=tf.float32),
                                               bias_initializer=tf.zeros_initializer(),
                                               data_format='channels_last')

    def __call__(self, inputs):
        return self.convt(inputs)


def masked_ce_loss(outputs, targets, lengths=None, mask=None, max_len=None):
    if lengths is None and mask is None:
        raise RuntimeError('Please provide either lengths or mask')

    # [batch_size, time_length]
    if mask is None:
        mask = sequence_mask(lengths, max_len, False)

    num_classes = tf.shape(outputs)[-1]
    # One hot encode targets (outputs.shape[-1] = hparams.quantize_channels)
    targets_ = tf.one_hot(tf.cast(targets, tf.int32), depth=tf.shape(outputs)[-1])

    targets_ = tf.reshape(targets_, [-1, num_classes])
    outputs = tf.reshape(outputs, [-1, num_classes])
    mask = tf.reshape(mask, [-1])
    with tf.control_dependencies([tf.assert_equal(tf.shape(outputs), tf.shape(targets_))]):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets_)

    with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
        masked_loss = losses * mask

    return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)


def masked_mixture_loss(outputs, targets, hparams, lengths=None, mask=None, max_len=None):
    if lengths is None and mask is None:
        raise RuntimeError('Please provide either lengths or mask')

        # [batch_size, time_length, 1]
    if mask is None:
        mask = sequence_mask(lengths, max_len, True)

        # [batch_size, time_length, dimension]
    ones = tf.ones([tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], tf.float32)
    mask_ = mask * ones

    losses = discretized_mix_logistic_loss(
        outputs, targets, num_classes=hparams.quantize_channels,
        log_scale_min=hparams.log_scale_min, reduce=False)

    with tf.control_dependencies([tf.assert_equal(tf.shape(losses), tf.shape(targets))]):
        return tf.reduce_sum(losses * mask_) / tf.reduce_sum(mask_)
