import os
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
from scipy.io import wavfile

import tensorflow as tf

import infolog
from wavenet_vocoder import util
from hparams import hparams
from wavenet_vocoder.models import create_model
from wavenet_vocoder.feeder import get_dataset
from tacotron.utils import ValueWindow

log = infolog.log


def add_stats(model, scope):
    with tf.variable_scope(scope):
        tf.summary.histogram('wav_outputs', model.y_hat)
        tf.summary.histogram('wav_targets', model.y)
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('learning_rate', model.learning_rate)
        return tf.summary.merge_all(scope=tf.get_variable_scope().name)


def add_test_stats(summary_writer, step, eval_loss):
    values = [
        tf.Summary.Value(tag='eval_stats/eval_loss', simple_value=eval_loss),
    ]
    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def create_shadow_saver(model, global_step):
    """Load shadow variables of saved model.

    Inspired by: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    Can also use: shadow_dict = model.ema.variables_to_restore()
    """
    # Add global step to saved variables to save checkpoints correctly
    shadow_variables = [model.ema.average_name(v) for v in model.variables]
    variables = model.variables

    if global_step is not None:
        shadow_variables += ['global_step']
        variables += [global_step]
    shadow_dict = dict(zip(shadow_variables, variables))  # dict(zip(keys, values)) -> {key1: value1, key2: value2, ...}
    return tf.train.Saver(shadow_dict, max_to_keep=5)


def load_averaged_model(sess, sh_saver, checkpoint_path):
    sh_saver.restore(sess, checkpoint_path)


def save_checkpoint(sess, saver, checkpoint_path, global_step):
    saver.save(sess, checkpoint_path, global_step=global_step)


def save_log(sess, global_step, model, plot_dir, audio_dir, hparams):
    log('Saving intermediate states at step {}'.format(global_step))
    bacth_y_hat, batch_y, lengths = sess.run([model.y_hat_log, model.y_log, model.input_lengths])
    idx = np.argmax(lengths)

    assert len(bacth_y_hat.shape) == 2
    assert len(batch_y.shape) == 2

    y_hat, y = bacth_y_hat[idx], batch_y[idx]
    if util.is_mulaw_quantize(hparams.input_type):
        y_hat = util.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y = util.inv_mulaw_quantize(y, hparams.quantize_channels)
    elif util.is_mulaw(hparams.input_type):
        y_hat = util.inv_mulaw(y_hat, hparams.quantize_channels)
        y = util.inv_mulaw(y, hparams.quantize_channels)

    # mask by length
    y_hat[lengths[idx]:] = 0
    y[lengths[idx]:] = 0

    # Make audio and plot paths
    pred_wav_path = os.path.join(audio_dir, 'step-{}-pred.wav'.format(global_step))
    target_wav_path = os.path.join(audio_dir, 'step-{}-real.wav'.format(global_step))
    plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))

    # Save audio
    librosa.output.write_wav(pred_wav_path, y_hat, sr=hparams.sample_rate)
    librosa.output.write_wav(target_wav_path, y, sr=hparams.sample_rate)

    # Save figure
    util.waveplot(plot_path, y_hat, y, hparams)




def eval_step(sess, global_step, model, plot_dir, audio_dir, summary_writer, hparams):
    """Evaluate model during training. Supposes that model variables are averaged.
    """
    start_time = time.time()
    y_hat, y_target, loss = sess.run([model.y_hat, model.y_target, model.eval_loss])
    duration = time.time() - start_time
    log('Time Evaluation: Generation of {} audio frames took {:.3f} sec ({:.3f} frames/sec)'.format(
        len(y_target), duration, len(y_target) / duration))
    print(y_hat)
    print(y_target)

    if util.is_mulaw_quantize(hparams.input_type):
        y_hat = util.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y_target = util.inv_mulaw_quantize(y_target, hparams.quantize_channels)
    elif util.is_mulaw(hparams.input_type):
        y_hat = util.inv_mulaw(y_hat, hparams.quantize_channels)
        y_target = util.inv_mulaw(y_target, hparams.quantize_channels)

    pred_wav_path = os.path.join(audio_dir, 'step-{}-pred.wav'.format(global_step))
    target_wav_path = os.path.join(audio_dir, 'step-{}-real.wav'.format(global_step))
    plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))

    # Save Audio
    wavfile.write(pred_wav_path, hparams.sample_rate, y_hat)
    wavfile.write(target_wav_path, hparams.sample_rate, y_target)

    # Save figure
    util.waveplot(plot_path, y_hat, y_target, hparams)
    log('Eval loss for global step {}: {:.3f}'.format(global_step, loss))

    log('Writing eval summary!')
    add_test_stats(summary_writer, global_step, loss)
    sess.run(model.clear_op)


def create_train_model(feeder, hp, global_step):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        model = create_model('WaveNet', hparams=hp)
        local_condition = None if feeder[3].dtype.is_bool else feeder[3]
        global_condition = None if feeder[4].dtype.is_bool else feeder[4]
        model.initialize(feeder[1], local_condition, global_condition, feeder[2], x=feeder[0])
        model.add_loss()
        model.add_optimizer(global_step)
        return model


def create_val_model(feeder, hp):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        model = create_model('WaveNet', hparams=hp)
        local_condition = None if feeder[3].dtype.is_bool else feeder[3]
        global_condition = None if feeder[4].dtype.is_bool else feeder[4]
        model.initialize(feeder[1], local_condition, global_condition, feeder[2], x=None)
        model.add_loss()
        return model


def train(log_dir, args, hp):
    # create dir
    save_dir = os.path.join(log_dir, 'checkpoints/')
    checkpoint_path = os.path.join(save_dir, 'wavenet_model.ckpt')
    audio_dir = os.path.join(log_dir, 'wavs')
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    eval_dir = os.path.join(log_dir, 'eval_dir')
    eval_audio_dir = os.path.join(eval_dir, 'wavs')
    eval_plot_dir = os.path.join(eval_dir, 'plots')
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_audio_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    # create dataset and iterator
    train_dataset = get_dataset(args.wavenet_train_file, shuffle=True)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    # feeder: inputs, targets, input_lengths, local_condition, global_condition
    feeder = iterator.get_next()
    train_init = iterator.make_initializer(train_dataset)

    # global step
    global_step = tf.Variable(-1, name='global_step', trainable=False)

    # create model
    train_model = create_train_model(feeder, hparams, global_step)
    val_model = create_val_model(feeder, hparams)

    # save stats
    train_stats = add_stats(train_model, 'train_stats')

    # book keeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    sh_saver = create_shadow_saver(train_model, global_step)

    # sess config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        for epoch in range(args.wavenet_train_epochs):

            # ================================== Train ================================== #

            # for each epoch, you have to init the iterator
            sess.run(train_init)

            while True:
                try:
                    start_time = time.time()
                    step, loss, _ = sess.run([global_step, train_model.loss, train_model.optimize])
                    time_window.append(time.time() - start_time)
                    loss_window.append(loss)
                    message = 'Epoch {:4d} Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
                        epoch, step, time_window.average, loss, loss_window.average)
                    log(message)

                    if loss > 100 or np.isnan(loss):
                        log('Loss exploded to {:.5f} at step {}'.format(loss, step))
                        raise Exception('Loss exploded')

                    if step % args.summary_interval == 0:
                        log('Writing summary at step {}'.format(step))
                        summary_writer.add_summary(sess.run(train_stats), step)

                    if step % args.checkpoint_interval == 0:
                        save_log(sess, step, train_model, plot_dir, audio_dir, hparams=hp)
                        save_checkpoint(sess, sh_saver, checkpoint_path, global_step)

                    if step % args.eval_interval == 0:
                        eval_step(sess, step, val_model, eval_plot_dir, eval_audio_dir, summary_writer, hparams)

                except tf.errors.OutOfRangeError:
                    break


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train_tacotron.log'), run_name)
    return log_dir, modified_hp


def main():
    parser = argparse.ArgumentParser(description='Train WaveNet')
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyper parameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--wavenet_train_file', default='training_data/train.txt')
    parser.add_argument('--wavenet_val_file', default='training_data/val.txt')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='wavenet')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')

    parser.add_argument('--restore_step', default=None, type=int, help='the restore step')

    parser.add_argument('--summary_interval', type=int, default=20,
                        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=200,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Steps between eval on test_func data')
    parser.add_argument('--wavenet_train_epochs', type=int, default=1000,
                        help='total number of tacotron training steps')
    parser.add_argument('--tf_log_level', type=int, default=2, help='Tensorflow C++ log level.')
    args = parser.parse_args()

    log_dir, hp = prepare_run(args)
    train(log_dir, args, hp)


if __name__ == '__main__':
    main()
