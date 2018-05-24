import tensorflow as tf

from tacotron.utils.symbols import symbols
from tacotron.models.module import EncoderConv, EncoderRNN, PreNet, DecoderRNN, FrameProjection, StopProjection, \
    Postnet, masked_mse, masked_sigmoid_ce
from tacotron.models.moudle_wrappers import TacotronEncoderWrapper, TacotronDecoderWrapper
from tacotron.models.attention import LocationSensitiveAttention
from tacotron.models.helper import TacoTestHelper, TacoTrainingHelper
from tacotron.models.custom_decoder import CustomDecoder


class Tacotron(object):

    def __init__(self, hparams):
        self.hp = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None, linear_targets=None,
                   target_lengths=None, gta=False, global_step=None, is_training=False, is_evaluating=False):
        """
        init tacotron for inference
        Args:
            inputs: int32 Tensor [B x T_in], the input of the model
            input_lengths: int32 Tensor [B], the length of each sentence in the input
            mel_targets: float32 Tensor [B x T_out x num_mels], the mel output, need for training
            stop_token_targets: int32 Tensor [B x T_out], the token decides the decoding
            linear_targets: float32 Tensor [B x T_out x num_freq], the linear output, need for training
            gta: whether use Ground Truth Aligned synthesis for WaveNet Vocoder
            target_lengths: the target lengths of mel, for GTA mode
            is_evaluating: eval mode
            is_training: train mode
            global_step: global step

        """
        if mel_targets is None and stop_token_targets is not None:
            raise ValueError('no mel targets were provided but token_targets were given')
        if mel_targets is not None and stop_token_targets is None and not gta:
            raise ValueError('Mel targets are provided without corresponding token_targets')
        if gta is False and self.hp.predict_linear is True and linear_targets is None:
            raise ValueError('Model is set to use post processing to predict linear '
                             'spectrograms in training but no linear targets given!')
        if gta and linear_targets is not None:
            raise ValueError('Linear spectrogram prediction is not supported in GTA mode!')
        if is_training and self.hp.mask_decoder and target_lengths is None:
            raise RuntimeError('Model set to mask paddings but no targets lengths provided for the mask!')
        if is_training and is_evaluating:
            raise RuntimeError('Model can not be in training and evaluation modes at the same time!')

        with tf.variable_scope('inference') as scope:
            is_training = mel_targets is not None and not gta
            batch_size = tf.shape(inputs)[0]

            assert self.hp.tacotron_teacher_forcing_mode in ('constant', 'scheduled')
            if self.hp.tacotron_teacher_forcing_mode == 'scheduled' and is_training:
                assert global_step is not None

            # GTA is only use for predict mel spectrogram to train WaveNet Vocoder
            post_condition = self.hp.predict_linear and not gta

            # =================================== Embedded =================================== #
            embedding_table = tf.get_variable(name='inputs_embedding', shape=[len(symbols), self.hp.embedding_dim],
                                              dtype=tf.float32)
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)  # [B, T_in, embedding_dim]

            # =================================== Encoder =================================== #
            # =====> Encoder Convolution
            encoder_conv = EncoderConv(is_training, hparams=self.hp, scope="encoder_conv")
            # =====> Encoder LSTM
            encoder_lstm = EncoderRNN(is_training, size=self.hp.encoder_lstm_units,
                                      zoneout=self.hp.tacotron_zoneout_rate, scope='encoder_LSTM')
            # =====> Combine Encoder Modules
            encoder = TacotronEncoderWrapper(conv_layer=encoder_conv,
                                             lstm_layer=encoder_lstm)
            # =====> Get Encoder outputs
            encoder_outputs = encoder(embedded_inputs, input_lengths)

            # =================================== Decoder =================================== #
            # =====> PreNet
            prenet = PreNet(is_training, layer_sizes=self.hp.prenet_layers, drop_rate=self.hp.tacotron_dropout_rate,
                            scope='decoder_prenet')
            # =====> Attention Mechanism(location sensitive attention)
            attention_mechanism = LocationSensitiveAttention(self.hp.attention_dim, encoder_outputs,
                                                             self.hp,
                                                             mask_encoder=self.hp.mask_encoder,
                                                             memory_sequence_length=input_lengths,
                                                             smoothing=self.hp.smoothing,
                                                             cumulate_weights=self.hp.cumulative_weights)
            # =====> Decoder LSTM
            decoder_lstm = DecoderRNN(is_training, layers=self.hp.decoder_layers,
                                      size=self.hp.decoder_lstm_units, zoneout=self.hp.tacotron_zoneout_rate,
                                      scope='decoder_lstm')
            # =====> Frames Projection Layer
            frame_projection = FrameProjection(self.hp.num_mels * self.hp.outputs_per_step, scope='linear_transform')
            # =====> Stop-Token Projection Layer
            stop_projection = StopProjection(is_training, self.hp.outputs_per_step, scope='stop_token_projection')
            # =====> Combine Decoder Modules

            decoder_cell = TacotronDecoderWrapper(
                prenet,
                attention_mechanism,
                decoder_lstm,
                frame_projection,
                stop_projection
            )

            # Define the helper for our decoder
            if is_training or is_evaluating or gta:
                self.helper = TacoTrainingHelper(batch_size, mel_targets, stop_token_targets, self.hp, gta,
                                                 is_evaluating,
                                                 global_step)
            else:
                self.helper = TacoTestHelper(batch_size, self.hp)

            # initial decoder state
            decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            # Only use max iterations at synthesis time
            max_iters = self.hp.max_iters if not (is_training or is_evaluating) else None

            # Decode
            (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                CustomDecoder(decoder_cell, self.helper, decoder_init_state),
                impute_finished=False,
                maximum_iterations=max_iters,
                swap_memory=self.hp.tacotron_swap_with_cpu)

            # Reshape outputs to be one output per entry
            # =====> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
            decoder_output = tf.reshape(frames_prediction, [batch_size, -1, self.hp.num_mels])
            stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

            # =================================== Post Net =================================== #
            post_net = Postnet(is_training, hparams=self.hp, scope='postnet_convolutions')
            # Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
            residual = post_net(decoder_output)
            # Project residual to same dimension as mel spectrogram
            # ==> [batch_size, decoder_steps * r, num_mels]
            residual_projection = FrameProjection(self.hp.num_mels, scope='postnet_projection')
            projected_residual = residual_projection(residual)

            # Compute the mel spectrogram
            mel_outputs = decoder_output + projected_residual

            if post_condition:
                # Based on https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
                # Post-processing Network to map mels to linear spectrograms using same architecture as the encoder
                post_processing_cell = TacotronEncoderWrapper(
                    EncoderConv(is_training, hparams=self.hp, scope='post_processing_convolutions'),
                    EncoderRNN(is_training, size=self.hp.encoder_lstm_units,
                               zoneout=self.hp.tacotron_zoneout_rate, scope='post_processing_LSTM'))

                expand_outputs = post_processing_cell(mel_outputs)
                linear_outputs = FrameProjection(self.hp.num_freq, scope='post_processing_projection')(expand_outputs)

            # Grab alignments from the final decoder state
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

            # save info at one step
            if is_training:
                self.ratio = self.helper.ratio

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.decoder_output = decoder_output
            self.alignments = alignments
            self.stop_token_prediction = stop_token_prediction
            self.stop_token_targets = stop_token_targets
            self.mel_outputs = mel_outputs

            if post_condition:
                self.linear_outputs = linear_outputs
                self.linear_targets = linear_targets
            self.mel_targets = mel_targets
            self.target_lengths = target_lengths

    def add_loss(self):
        """Adds loss to the model. Sets "loss" field. initialize must have been called."""
        with tf.variable_scope('loss'):
            if self.hp.mask_decoder:
                # Compute loss of predictions before postnet
                before = masked_mse(self.mel_targets, self.decoder_output, self.target_lengths, self.hp)
                # Compute loss after postnet
                after = masked_mse(self.mel_targets, self.mel_outputs, self.target_lengths, self.hp)
                # Compute <stop_token> loss (for learning dynamic generation stop)
                stop_token_loss = masked_sigmoid_ce(self.stop_token_targets,
                                                    self.stop_token_prediction, self.target_lengths,
                                                    hparams=self.hp)
            else:
                # Compute loss of predictions before postnet
                before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
                # Compute loss after postnet
                after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
                # Compute <stop_token> loss (for learning dynamic generation stop)
                stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.stop_token_targets,
                    logits=self.stop_token_prediction))

            if self.hp.predict_linear:
                # Compute linear loss
                # From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
                # Prioritize loss for frequencies under 2000 Hz.
                l1 = tf.abs(self.linear_targets - self.linear_outputs)
                n_priority_freq = int(2000 / (self.hp.sample_rate * 0.5) * self.hp.num_freq)
                linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
            else:
                linear_loss = 0.

            # Compute the regularization weight
            if self.hp.tacotron_scale_regularization:
                reg_weight_scaler = 1. / (2 * self.hp.max_abs_value) if self.hp.symmetric_mels else 1. / (
                    self.hp.max_abs_value)
                reg_weight = self.hp.tacotron_reg_weight * reg_weight_scaler
            else:
                reg_weight = self.hp.tacotron_reg_weight

            # Get all trainable variables
            all_vars = tf.trainable_variables()
            regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
                                       if not ('bias' in v.name or 'Bias' in v.name)]) * reg_weight

            # Compute final loss term
            self.before_loss = before
            self.after_loss = after
            self.stop_token_loss = stop_token_loss
            self.regularization_loss = regularization
            self.linear_loss = linear_loss

            self.loss = self.before_loss + self.after_loss + self.stop_token_loss \
                        + self.regularization_loss + self.linear_loss

    def add_optimizer(self, global_step):
        """Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
            Args:
                global_step: int32 scalar Tensor representing current global step in training
        """
        with tf.variable_scope('optimizer') as scope:
            hp = self.hp
            if hp.tacotron_decay_learning_rate:
                self.decay_steps = hp.tacotron_decay_steps
                self.decay_rate = hp.tacotron_decay_rate
                self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
                                               hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            # Just for caution
            # https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:

        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps

        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k

        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self.hp

        # Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr,
                                        global_step - hp.tacotron_start_decay,  # lr = 1e-3 at step 50k
                                        self.decay_steps,
                                        self.decay_rate,  # lr = 1e-5 around step 310k
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)
