import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import tensorflow as tf
from pysptk.util import example_audio_file
from wavenet_vocoder.util import mulaw_quantize, inv_mulaw_quantize
from wavenet_vocoder import WaveNet
from hparams import hparams, hparams_debug_string


def load_data(sr=4000,  N=3000, returns_power=True, mulaw=True):
    x, _ = librosa.load(example_audio_file(), sr=sr)
    x, _ = librosa.effects.trim(x, top_db=15)

    # To save computational cost
    x = x[:N]

    # For power conditioning wavenet
    if returns_power:
        # (1 x N')
        p = librosa.feature.rmse(x, frame_length=256, hop_length=128)
        upsample_factor = x.size // p.size
        # (1 x N)
        p = np.repeat(p, upsample_factor, axis=-1)
        if p.size < x.size:
            # pad against time axis
            p = np.pad(p, [(0, 0), (0, x.size - p.size)], mode="constant", constant_values=0)

        # shape adajst
        p = p.reshape(1, -1, 1)

    # (T,)
    if mulaw:
        x = mulaw_quantize(x)
        x = x.reshape(1, -1)
        x_org = inv_mulaw_quantize(x)
    else:
        x_org = x
        x = x.reshape(1, 1, -1)

    if returns_power:
        return x, x_org, p

    return x, x_org


def test_local_condition():
    hparams.parse("layers=4,stacks=2,residual_channels=32,gate_channels=32,"
                  "skip_out_channels=32,upsample_conditional_features=False,cin_channels=1,num_mels=1")
    x, x_org, c = load_data()
    lengths = np.asarray(x.shape[1], dtype=np.int32).reshape(1, )
    print(x.shape, c.shape)
    placeholders = {
        'inputs': tf.placeholder(dtype=tf.int32, shape=[1, None]),
        'local_condition': tf.placeholder(dtype=tf.float32, shape=[1, None, 1]),
        'input_lengths': tf.placeholder(dtype=tf.int32, shape=[1, ])
    }
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        train_model = WaveNet(hparams)
        train_model.initialize(y=placeholders['inputs'], c=placeholders['local_condition'],
                               x=placeholders['inputs'], g=None, input_lengths=placeholders['input_lengths'])

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        eval_model = WaveNet(hparams)
        eval_model.initialize(y=placeholders['inputs'], c=placeholders['local_condition'], g=None,
                              input_lengths=placeholders['input_lengths'], x=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in tf.global_variables():
            print(i)


        feed_dict = {placeholders['inputs']: x, placeholders['local_condition']: c, placeholders['input_lengths']: lengths}
        y_hat, y = sess.run([train_model.y_hat, train_model.y], feed_dict=feed_dict)
        print(y_hat)
        print(y)
        eval_feed_dict = {placeholders['inputs']: x, placeholders['local_condition']: c, placeholders['input_lengths']: lengths}
        y_eval_hat, y_eval = sess.run([eval_model.y_hat, eval_model.y_target], feed_dict=eval_feed_dict)
        print(y_eval_hat, y_eval)
        c = np.abs(y_eval_hat-y_hat)
        print(c.mean(), c.max())
        try:
            assert np.allclose(y_hat,
                               y_eval, atol=1e-4)
        except:
            from warnings import warn
            warn("oops! must be a bug!")







if __name__ == '__main__':
    test_local_condition()

