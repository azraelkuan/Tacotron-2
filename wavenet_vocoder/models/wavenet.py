import numpy as np
import tensorflow as tf

from wavenet_vocoder.models.module import Conv1d1x1, ResidualConv1dGLU, ConvTransposed2d, \
    Embedding, masked_ce_loss, masked_mixture_loss, ReLU
from wavenet_vocoder.models.mixture import sample_from_discretized_mix_logistic
from wavenet_vocoder import util

from infolog import log
from datasets import audio


def _expand_global_features(batch_size, time_length, global_features, data_format='BTC'):
    """Expand global condition to all time steps

    Args:
        batch_size: int
        time_length: int
        global_features: b x 1 x c
        data_format: the output format

    Returns:
        None or Tensor of shape [B T C]
    """
    accepted_formats = ['BCT', 'BTC']
    if not (data_format in accepted_formats):
        raise ValueError('{} is an unknow data format, accepted formats are "BCT" and "BTC"'.format(data_format))

    if global_features is None:
        return None

    # [batch_size, channels] ==> [batch_size, channels, 1]
    g = tf.cond(tf.equal(tf.rank(global_features), 2),
                lambda: tf.expand_dims(global_features, axis=-1),
                lambda: global_features)

    # [batch_size, channels, 1] ==> [batch_size, channels, time_length]
    g = tf.tile(g, [1, 1, time_length])

    if data_format == 'BTC':
        # [batch_size, time_length, channels]
        return tf.transpose(g, [0, 2, 1])
    else:
        return g


def receptive_field_size(total_layers, stacks, kernel_size, func=lambda x: 2 ** x):
    """Compute receptive file size."""
    assert total_layers % stacks == 0
    layers_per_stack = total_layers // stacks
    dilations = [func(i % layers_per_stack) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) - 1


class WaveNet(object):

    def __init__(self, hparams):
        """Construct the WaveNet Architecture"""
        self.hp = hparams

        if self.local_conditioning_enabled():
            assert hparams.num_mels == hparams.cin_channels

        # Initialize model architecture
        assert hparams.layers % hparams.stacks == 0
        layers_per_stack = hparams.layers // hparams.stacks

        # determine the input type
        self.scalar_input = util.is_scalar_input(hparams.input_type)

        # first convolution

        self.first_conv = Conv1d1x1(filters=hparams.residual_channels, name='input_convolution')

        # Residual convolutions
        self.conv_layers = []
        for layer in range(hparams.layers):
            self.conv_layers.append(
                ResidualConv1dGLU(
                    residual_channels=hparams.residual_channels,
                    gate_channels=hparams.gate_channels,
                    kernel_size=hparams.kernel_size,
                    skip_out_channels=hparams.skip_out_channels,
                    use_bias=hparams.use_bias,
                    dilation=2 ** (layer % layers_per_stack),
                    dropout=hparams.wavenet_dropout,
                    cin_channels=hparams.cin_channels,
                    gin_channels=hparams.gin_channels,
                    scope='ResidualConv1dGLU_{}'.format(layer),
                )
            )

        # Skip convolutions

        self.last_conv_layers = [
            ReLU(name='final_relu_1'),
            Conv1d1x1(hparams.skip_out_channels, name='final_conv_1'),
            ReLU(name='finaL_relu_2'),
            Conv1d1x1(hparams.out_channels, name="final_conv_2")
        ]

        # Global condition embedding
        if hparams.gin_channels > 0 and hparams.use_speaker_embedding:
            assert hparams.n_speakers is not None
            self.embed_speakers = Embedding(hparams.n_speakers, hparams.gin_channels, std=0.1, name='gc_embedding')
        else:
            self.embed_speakers = None

        # Upsample conv
        if hparams.upsample_conditional_features:
            self.upsample_conv = []
            for i, s in enumerate(hparams.upsample_scales):
                    convt = ConvTransposed2d(1, s, hparams.freq_axis_kernel_size, padding='same', strides=(s, 1),
                                             scope='local_conditioning_upsample_{}'.format(i + 1))
                    self.upsample_conv.append(convt)
        else:
            self.upsample_conv = None

        if hparams.upsample_conditional_features:
            self.all_convs = [self.first_conv] + self.conv_layers + self.last_conv_layers + self.upsample_conv
        else:
            self.all_convs = [self.first_conv] + self.conv_layers + self.last_conv_layers

        self.receptive_field = receptive_field_size(hparams.layers, hparams.stacks, hparams.kernel_size)

    # Sanity check functions
    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.hp.cin_channels > 0

    def set_mode(self, is_training):
        for conv in self.all_convs:
            try:
                conv.set_mode(is_training)
            except AttributeError:
                pass

    def initialize(self, y, c, g, input_lengths, x=None, synthesis_length=None):
        """ Initialize wavenet graph for train, eval and test_func cases.

        Args:
            y: target [B x T]
            c: None or B x T x C
            g: None or [B]
            input_lengths:
            x: [B x T] Tensor
            synthesis_length:

        Returns:

        """
        self.is_training = x is not None
        self.is_evaluating = not self.is_training and y is not None
        # set all conv to the correspond mode
        self.set_mode(self.is_training)

        log('Initializing Wavenet model.  Dimensions (? = dynamic shape): ')
        log('  Train mode:                {}'.format(self.is_training))
        log('  Eval mode:                 {}'.format(self.is_evaluating))
        log('  Synthesis mode:            {}'.format(not (self.is_training or self.is_evaluating)))

        if self.is_training:
            if util.is_mulaw_quantize(self.hp.input_type):
                x = tf.one_hot(tf.cast(x, tf.int32), self.hp.quantize_channels)
            else:
                x = tf.expand_dims(x, axis=-1)
            tf.assert_equal(tf.rank(x), 3, name='assert_x_shape')

        with tf.variable_scope("inference"):
            # =========================== Training =========================== #
            if self.is_training:
                batch_size = tf.shape(x)[0]
                # [B T 1]
                self.mask = self.get_mask(input_lengths, maxlen=tf.shape(x)[1])
                # [B T C]
                y_hat = self.step(x, c, g, softmax=False)

                self.y_hat = y_hat
                self.y = y
                self.input_lengths = input_lengths

                # for training log
                shape_control = (batch_size, tf.shape(x)[-1])
                with tf.control_dependencies([tf.assert_equal(tf.shape(y), shape_control)]):
                    y_log = y
                    if util.is_raw(self.hp.input_type):
                        self.y = tf.expand_dims(y, axis=-1)

                y_hat_log = y_hat
                # [B T C]
                y_hat_log = tf.reshape(y_hat_log, [batch_size, -1, self.hp.out_channels])

                if util.is_mulaw_quantize(self.hp.input_type):
                    # [batch_size, time_length]
                    y_hat_log = tf.argmax(tf.nn.softmax(y_hat_log, -1), -1)

                else:
                    # [batch_size, time_length]
                    y_hat_log = sample_from_discretized_mix_logistic(
                        y_hat_log, log_scale_min=self.hp.log_scale_min)

                self.y_log = y_log
                self.y_hat_log = y_hat_log

            elif self.is_evaluating:
                test_inputs = tf.one_hot(tf.cast(y, tf.int32), self.hp.quantize_channels)

                # eval one sentence use incremental forward for testing
                idx = 0
                length = input_lengths[idx]
                y_target = tf.reshape(y[idx], [-1])[:length]  # [T, ]

                if c is not None:
                    if self.hp.upsample_conditional_features:
                        c = tf.expand_dims(c[idx, :length//audio.get_hop_size(self.hp), :], axis=0)
                    else:
                        c = tf.expand_dims(c[idx, :length, :], axis=0)
                    # [1 T_c C]
                    with tf.control_dependencies([tf.assert_equal(tf.rank(c), 3)]):
                        c = tf.identity(c, name='eval_assert_c_rank_op')

                if g is not None:
                    g = g[idx]  # [1, ]

                if util.is_mulaw_quantize(self.hp.input_type):
                    initial_value = util.mulaw_quantize(0, self.hp.quantize_channels)
                elif util.is_mulaw(self.hp.input_type):
                    initial_value = util.mulaw(0, self.hp.quantize_channels)
                else:
                    initial_value = 0.

                print('eval initial value: {}'.format(initial_value))

                # initial input [1 1 C_initial]
                if util.is_mulaw_quantize(self.hp.input_type):
                    initial_input = tf.one_hot(initial_value, depth=self.hp.quantize_channels)
                    initial_input = tf.reshape(initial_input, [1, 1, self.hp.quantize_channels])
                else:
                    initial_input = tf.ones([1, 1, 1], dtype=tf.float32) * initial_value

                # fast generation
                y_hat = self.incremental(initial_input, c=c, g=g, time_length=length,
                                         softmax=True, quantize=True, log_scale_min=self.hp.log_scale_min)

                if util.is_mulaw_quantize(self.hp.input_type):
                    self.y_eval = tf.reshape(y[idx], [1, -1])[:, :length]
                else:
                    self.y_eval = tf.expand_dims(y[idx], 0)[:, :length, :]
                self.eval_length = length

                if util.is_mulaw_quantize(self.hp.input_type):
                    # y_hat [B x T x C]
                    y_hat = tf.reshape(tf.argmax(y_hat, axis=-1), [-1])
                else:
                    y_hat = tf.reshape(y_hat, [-1])

                self.y_hat = y_hat
                self.y_target = y_target

        # apply ema to variable
        self.variables = tf.trainable_variables()
        self.ema = tf.train.ExponentialMovingAverage(decay=self.hp.wavenet_ema_decay)

    def get_mask(self, input_lengths, maxlen=None):
        expand = not util.is_mulaw_quantize(self.hp.input_type)
        mask = util.sequence_mask(input_lengths, max_len=maxlen, expand=expand)

        if util.is_mulaw_quantize(self.hp.input_type):
            return mask[:, 1:]
        return mask[:, 1:, :]

    def step(self, x, c=None, g=None, softmax=False):
        """Forward step
            Args:
                x: Tensor of shape [batch_size, time_length, channels], One-hot encoded audio signal.
                c: Tensor of shape [batch_size, time_length, cin_channels], Local conditioning features.
                g: Tensor of shape [batch_size, 1, gin_channels] or Ids of shape [batch_size, 1],
                    Global conditioning features.
                    Note: set hparams.use_speaker_embedding to False to disable embedding layer and
                    use extrnal One-hot encoded features.
                softmax: Boolean, Whether to apply softmax.
            Returns:
                a Tensor of shape [batch_size, out_channels, time_length]
        """
        batch_size = tf.shape(x)[0]
        time_length = tf.shape(x)[1]
        # prepare global condition
        if g is None:
            if self.embed_speakers is not None:
                # [B 1] ==> [B 1 gin_channels]
                g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))

        # expose global condition to all time step ==> [B T gin_channels]
        g_btc = _expand_global_features(batch_size, time_length, g, data_format='BTC')

        # prepare local condition [B T cin_channels] ==> [B new_T cin_channels]
        if c is not None and self.upsample_conv is not None:
            c = tf.expand_dims(c, axis=-1)  # [B T cin_channels 1]
            for transposed_conv in self.upsample_conv:
                c = transposed_conv(c)
            c = tf.squeeze(c, axis=-1)  # [B new_T cin_channels]
            with tf.control_dependencies([tf.assert_equal(tf.shape(c)[1], tf.shape(x)[1])]):
                c = tf.identity(c, name='control_c_and_x_shape')

        # feed data
        x = self.first_conv(x)
        skips = None
        for conv in self.conv_layers:
            x, h = conv(x, c, g_btc)
            if skips is None:
                skips = h
            else:
                skips = skips + h

        x = skips
        for conv in self.last_conv_layers:
            x = conv(x)

        return tf.nn.softmax(x, axis=-1) if softmax else x

    def incremental(self, initial_input, c=None, g=None, time_length=100, test_inputs=None,
                    softmax=True, quantize=True, log_scale_min=-7.0):
        """ Incremental forward

        Args:
            initial_input: Tensor [B 1 C]
            c: None or Tensor [B T C]
            g: None or Tensor [B T C]
            time_length: int
            test_inputs: Tensor, teacher forcing inputs, for debug
            softmax: Boolean, whether apply softmax function
            quantize: whether to quantize softmax output before feeding to next time step
            log_scale_min: float, log scale minimum value

        Returns:
            Tensor of shape [B T C] or [B 1 C]
        """
        self.clear_buffer()
        batch_size = 1
        if test_inputs is not None:
            batch_size = tf.shape(test_inputs)[0]
            if time_length is None:
                time_length = tf.shape(test_inputs)[1]
            else:
                time_length = tf.maximum(time_length, tf.shape(test_inputs)[1])

        # global condition
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))

        # [B T C]
        self.g_btc = _expand_global_features(batch_size, time_length, g, data_format='BTC')

        # local condition
        if c is not None and self.upsample_conv is not None:
            c = tf.expand_dims(c, axis=-1)  # [B T C 1]
            for upsample_conv in self.upsample_conv:
                c = upsample_conv(c)
            c = tf.squeeze(c, axis=-1)
            tf.assert_equal(tf.shape(c)[1], time_length)

        self.c = c

        # initialize loop vars
        initial_time = tf.constant(0, dtype=tf.int32)
        if test_inputs is not None:
            initial_input = tf.expand_dims(test_inputs[:, 0, :], axis=1)

        initial_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        initial_debug_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        def condition(time, initial_input, initial_outputs_ta, initial_debug_outputs_ta):
            return tf.less(time, time_length)

        def body(time, current_input, outputs_ta, debug_outputs_ta):
            # condition features for single time step
            current_c = None if self.c is None else tf.expand_dims(self.c[:, time, :], axis=1)
            current_g = None if self.g_btc is None else tf.expand_dims(self.g_btc[:, time, :], axis=1)

            x = self.first_conv.incremental_forward(current_input)
            skips = None
            for conv in self.conv_layers:
                x, h = conv.incremental_forward(x, current_c, current_g, scope=conv.scope)
                skips = h if skips is None else (skips + h)
            x = skips
            for conv in self.last_conv_layers:
                try:
                    x = conv.incremental_forward(x)
                except AttributeError:
                    x = conv(x)
            # x [1, 1, 256]
            debug_outputs_ta = debug_outputs_ta.write(time, tf.squeeze(x, axis=1))  # squeeze time dim

            if self.scalar_input:
                x = sample_from_discretized_mix_logistic(tf.reshape(x, [batch_size, -1, 1]),
                                                         log_scale_min=log_scale_min)
            else:
                # [1 256]
                x = tf.nn.softmax(tf.reshape(x, [batch_size, -1]), axis=1) \
                    if softmax else tf.reshape(x, [batch_size, -1])
                if quantize:
                    # sample = tf.multinomial(x, 1)[0]  # [1, ] only one sample
                    sample = tf.py_func(np.random.choice,
                                        [np.arange(self.hp.out_channels), 1, True, tf.reshape(x, [-1])], tf.int64)
                    sample = tf.reshape(sample, [1, ])
                    # [1, 256],
                    x = tf.one_hot(sample, depth=self.hp.quantize_channels)

            outputs_ta = outputs_ta.write(time, x)

            time = time + 1
            if test_inputs is not None:
                next_input = tf.cond(tf.less(time, tf.shape(test_inputs)[1]),
                                     lambda: tf.expand_dims(test_inputs[:, time, :], axis=1),
                                     lambda: initial_input)
            else:
                # [1, 1, 256]
                next_input = tf.expand_dims(x, axis=1)

            return time, next_input, outputs_ta, debug_outputs_ta

        result = tf.while_loop(condition,
                               body,
                               loop_vars=[initial_time, initial_input, initial_outputs_ta, initial_debug_outputs_ta],
                               parallel_iterations=32,
                               swap_memory=self.hp.wavenet_swap_with_cpu)
        outputs_ta = result[2]
        # [T B C]
        outputs = outputs_ta.stack()
        eval_outputs = result[-1].stack()

        # [B T C]
        outputs = tf.transpose(outputs, [1, 0, 2])
        eval_outputs = tf.transpose(eval_outputs, [1, 0, 2])

        self.y_hat_eval = eval_outputs  # for debug

        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def add_loss(self):
        """Adds loss computation to the graph. Supposes that initialize function has already been called."""
        with tf.variable_scope('loss'):
            if self.is_training:
                if util.is_mulaw_quantize(self.hp.input_type):
                    self.loss = masked_ce_loss(self.y_hat[:, :-1, :], self.y[:, 1:], mask=self.mask)
                else:
                    self.loss = masked_mixture_loss(self.y_hat[:, :, :-1], self.y[:, 1:, :],
                                                    hparams=self.hp, mask=self.mask)
            elif self.is_evaluating:
                if util.is_mulaw_quantize(self.hp.input_type):
                    self.eval_loss = masked_ce_loss(self.y_hat_eval, self.y_eval, lengths=[self.eval_length])
                else:
                    self.eval_loss = masked_mixture_loss(self.y_hat_eval, self.y_eval, hparams=self.hp,
                                                         lengths=[self.eval_length])

    def add_optimizer(self, global_step):
        """Adds optimizer to the graph. Supposes that initialize function has already been called."""
        with tf.variable_scope('optimizer'):
            hp = self.hp

            if hp.wavenet_decay_learning_rate:
                self.learning_rate = _learning_rate_decay(hp.wavenet_init_lr, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.wavenet_init_lr)

            # Adam with constant learning rate
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.wavenet_adam_beta1,
                                               hp.wavenet_adam_beta2, hp.wavenet_adam_epsilon)

            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients

            # Gradients clipping
            if hp.wavenet_clip_gradient:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, hp.wavenet_clip_thresh)
            else:
                clipped_gradients = gradients

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

        # Add exponential moving average
        # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        # Use adam optimization process as a dependency
        with tf.control_dependencies([adam_optimize]):
            # Create the shadow variables and add ops to maintain moving averages
            # Also updates moving averages after each update step
            # This is the optimize call instead of traditional adam_optimize one.
            assert tuple(self.variables) == variables  # Verify all trainable variables are being averaged
            self.optimize = self.ema.apply(variables)


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
