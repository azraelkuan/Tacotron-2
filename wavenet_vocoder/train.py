import os
import time
import argparse
import librosa
import numpy as np

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
        return tf.summary.merge_all()


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


def train(log_dir, args, hp):
    # create dir
    save_dir = os.path.join(log_dir, 'checkpoints/')
    checkpoint_path = os.path.join(save_dir, 'wavenet_model.ckpt')
    audio_dir = os.path.join(log_dir, 'wavs')
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    # create dataset and iterator
    train_dataset = get_dataset('/mnt/lustre/sjtu/users/kc430/data/my/tacotron2/ljspeech/train.txt', shuffle=True)
    iterator = train_dataset.make_initializable_iterator()
    inputs, targets, input_lengths, local_conditions, global_conditions = iterator.get_next()

    # # global step
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    #
    # # create model
    # with tf.variable_scope('model'):
    #     model = create_model(name='WaveNet', hparams=hp)
    #     # extra inputs
    #
    #     model.
    #     model.add_loss()
    #     model.add_optimizer(global_step)
    #
    # # save stats
    # train_stats = add_stats(model, 'train')
    #
    # # book keeping
    # time_window = ValueWindow(100)
    # loss_window = ValueWindow(100)
    # sh_saver = create_shadow_saver(model, global_step)
    #
    # # sess config
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    #
    #     for epoch in range(args.wavenet_train_epochs):
    #         # for each epoch, you have to init the iterator
    #         sess.run(iterator.initializer)
    #
    #         while True:
    #             try:
    #                 start_time = time.time()
    #                 step, loss, _ = sess.run([global_step, model.loss, model.optimize])
    #                 time_window.append(time.time() - start_time)
    #                 loss_window.append(loss)
    #                 message = 'Epoch {:4d} Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
    #                     epoch, step, time_window.average, loss, loss_window.average)
    #                 log(message)
    #
    #                 if loss > 100 or np.isnan(loss):
    #                     log('Loss exploded to {:.5f} at step {}'.format(loss, step))
    #                     raise Exception('Loss exploded')
    #
    #                 if step % args.summary_interval == 0:
    #                     log('Writing summary at step {}'.format(step))
    #                     train_writer.add_summary(sess.run(train_stats), step)
    #
    #                 if step % args.checkpoint_interval == 0:
    #                     save_log(sess, step, model, plot_dir, audio_dir, hparams=hp)
    #                     save_checkpoint(sess, sh_saver, checkpoint_path, global_step)
    #
    #             except tf.errors.OutOfRangeError:
    #                 break


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

    parser.add_argument('--summary_interval', type=int, default=250,
                        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5000,
                        help='Steps between eval on test data')
    parser.add_argument('--wavenet_train_epochs', type=int, default=200,
                        help='total number of tacotron training steps')
    parser.add_argument('--tf_log_level', type=int, default=3, help='Tensorflow C++ log level.')
    args = parser.parse_args()

    log_dir, hp = prepare_run(args)
    train(log_dir, args, hp)


if __name__ == '__main__':
    main()
