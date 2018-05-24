import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

import tensorflow as tf

import infolog
from datasets import audio
from tacotron.feeder import get_dataset
from tacotron.models import create_model
from tacotron.utils import ValueWindow, plot
from hparams import hparams

log = infolog.log


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def add_stats(model, hp, scope):
    with tf.variable_scope(scope):
        tf.summary.histogram('mel_outputs', model.mel_outputs)
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('before_loss', model.before_loss)
        tf.summary.scalar('after_loss', model.after_loss)
        if hp.predict_linear:
            tf.summary.scalar('linear_loss', model.linear_loss)
        tf.summary.scalar('regularization_loss', model.regularization_loss)
        tf.summary.scalar('stop_token_loss', model.stop_token_loss)
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('learning_rate', model.learning_rate)  # Control learning rate decay speed
        # Control teacher forcing ratio decay when mode = 'scheduled'
        if hp.tacotron_teacher_forcing_mode == 'scheduled':
            tf.summary.scalar('teacher_forcing_ratio', model.ratio)
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        # visualize gradients (in case of explosion)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
        return tf.summary.merge_all()


def train(log_dir, args, hp):
    # prepare dir
    save_dir = os.path.join(log_dir, 'checkpoints/')
    checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    mel_dir = os.path.join(log_dir, 'mel-spectrograms')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)

    if hparams.predict_linear:
        linear_dir = os.path.join(log_dir, 'linear-spectrograms')
        os.makedirs(linear_dir, exist_ok=True)

    # get train data
    train_dataset = get_dataset(meta_file=args.tacotron_train_file, shuffle=True)
    # get val data
    val_dataset = get_dataset(meta_file=args.tacotron_val_file, shuffle=False)
    # define iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    # get next will return 6 value as below
    inputs, mel_targets, token_targets, linear_targets, input_lengths, target_lengths \
        = iterator.get_next()
    train_init = iterator.make_initializer(train_dataset)
    val_init = iterator.make_initializer(val_dataset)

    # define global step
    global_step = tf.Variable(-1, name='global_step', trainable=False)
    global_val_step = 0

    # define model
    with tf.variable_scope('model'):
        model = create_model('Tacotron', hp)
        if hp.predict_linear:
            model.initialize(inputs, input_lengths, mel_targets, token_targets, linear_targets,
                             target_lengths=target_lengths, global_step=global_step, is_training=True)
        else:
            model.initialize(inputs, input_lengths, mel_targets, token_targets, target_lengths=target_lengths,
                             global_step=global_step, is_training=True)
        model.add_loss()
        model.add_optimizer(global_step)
        train_stats = add_stats(model, hp, 'train_stats')
        val_stats = add_stats(model, hp, 'val_stats')

    # Book keeping
    train_time_window = ValueWindow(100)
    train_loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    # add gpu config for memory use
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # define tf summary writer
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # init all the variable
        sess.run(tf.global_variables_initializer())

        for epoch in range(args.tacotron_train_epochs):

            # ================================== Train Process ================================== #

            # for every epoch, need to init the iterator
            sess.run(train_init)
            debug = True

            while True:
                try:
                    start_time = time.time()
                    if debug:
                        step, loss, _, mel_p, mel_t, t_len, align = sess.run([global_step, model.loss, model.optimize,
                                                                              model.mel_outputs[0],
                                                                              model.mel_targets[0],
                                                                              model.target_lengths[0],
                                                                              model.alignments[0]])
                        debug = False
                    else:
                        step, loss, _ = sess.run([global_step, model.loss, model.optimize])
                    train_time_window.append(time.time() - start_time)
                    train_loss_window.append(loss)

                    message = 'Epoch {:3d} Step {:7d} [{:.3f} sec/Step, Train_Loss={:.5f}, Avg_Loss={:.5f}]'.format(
                        epoch, step, train_time_window.average, loss, train_loss_window.average)

                    log(message)

                    if loss > 1000 or np.isnan(loss):
                        log('Loss exploded to {:.5f} at step {}'.format(loss, step))
                        raise Exception('Loss exploded')

                    if step % args.summary_interval == 0:
                        stats = sess.run(train_stats)

                        log('Writing summary at step {}'.format(step))
                        summary_writer.add_summary(stats, step)

                    if step % args.checkpoint_interval == 0:
                        # Save model and current global step
                        saver.save(sess, checkpoint_path, global_step=global_step)

                    sys.stdout.flush()
                except tf.errors.OutOfRangeError:
                    # save status
                    save_debug_info(step, mel_p, mel_t, t_len, align, loss, wav_dir, plot_dir, args, mode='val')

                    log('Begin next val epoch')
                    break

            # ================================== Val Process ================================== #

            # for every epoch, need to init the iterator
            sess.run(val_init)
            debug = True
            val_loss = []
            while True:
                try:
                    if debug:
                        loss, mel_p, mel_t, t_len, align = sess.run(
                            [model.loss, model.mel_outputs[0], model.mel_targets[0],
                             model.target_lengths[0], model.alignments[0]])
                    else:
                        loss = sess.run(model.loss)

                    val_loss.append(loss)

                    if global_val_step % args.summary_interval == 0:
                        log('Writing summary at step {}'.format(step))
                        summary_writer.add_summary(sess.run(val_stats), step)

                    global_val_step += 1
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError:
                    avg_loss = sum(val_loss) / len(val_loss)
                    # save status
                    # save_debug_info(step, mel_p, mel_t, t_len, align, avg_loss, wav_dir, plot_dir, args, mode='val')

                    # log val loss
                    message = '==========> Epoch {:3d} Val_Avg_Loss={:.5f}] <=========='.format(
                        epoch, avg_loss)
                    log(message)

                    log('Begin next train epoch')
                    break


def save_debug_info(step, mel_p, mel_t, t_len, align, loss, wav_dir, plot_dir, args, mode='train'):
    """save audio and alignment."""
    wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
    audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-{}-waveform-mel.wav'.format(step, mode)),
                   sr=hparams.sample_rate)
    plot.plot_alignment(align, os.path.join(plot_dir, 'step-{}-{}-align.png'.format(step, mode)),
                        info='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step,
                                                                   loss),
                        max_len=t_len // hparams.outputs_per_step)
    plot.plot_spectrogram(mel_p,
                          os.path.join(plot_dir, 'step-{}-{}-mel-spectrogram.png'.format(step, mode)),
                          info='{}, {}, step={}, loss={:.5}'.format(args.model, time_string(), step,
                                                                    loss), target_spectrogram=mel_t,
                          max_len=t_len)


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train_tacotron.log'), run_name)
    return log_dir, modified_hp


def main():
    parser = argparse.ArgumentParser(description='Train Tacotron')
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyper parameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--tacotron_train_file', default='training_data/train.txt')
    parser.add_argument('--tacotron_val_file', default='training_data/val.txt')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='tacotron2')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')

    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=250,
                        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5000,
                        help='Steps between eval on test data')
    parser.add_argument('--tacotron_train_epochs', type=int, default=200,
                        help='total number of tacotron training steps')
    parser.add_argument('--tf_log_level', type=int, default=3, help='Tensorflow C++ log level.')
    args = parser.parse_args()

    log_dir, hp = prepare_run(args)
    train(log_dir, args, hp)


if __name__ == '__main__':
    main()
