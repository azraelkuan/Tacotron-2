import numpy as np
import tensorflow as tf
import librosa.display as dsp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _assert_valid_input_type(s):
    assert s == 'mulaw-quantize' or s == 'mulaw' or s == 'raw'


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == 'mulaw-quantize'


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == 'mulaw'


def is_raw(s):
    _assert_valid_input_type(s)
    return s == 'raw'


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)


# From https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/preprocessing/generic.py
def mulaw(x, mu=256):
    """Mu-Law companding"""
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def inv_mulaw(y, mu=256):
    """Inverse of mu-law companding (mu-law expansion)"""
    return _sign(y) * (1.0 / mu) * ((1.0 + mu) ** _abs(y) - 1.0)


def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize"""
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)


def inv_mulaw_quantize(y, mu=256):
    """Inverse of mu-law companding + quantize"""
    y = 2 * _asfloat(y) / mu - 1
    return inv_mulaw(y, mu)


def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else x.sign()


def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else x.log1p()


def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else x.abs()


def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()


def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()


def sequence_mask(input_lengths, max_len=None, expand=True):
    if max_len is None:
        max_len = tf.reduce_max(input_lengths)

    if expand:
        return tf.expand_dims(tf.sequence_mask(input_lengths, max_len, dtype=tf.float32), axis=-1)
    return tf.sequence_mask(input_lengths, max_len, dtype=tf.float32)


def waveplot(path, y_hat, y_target, hparams):
    sr = hparams.sample_rate

    plt.figure(figsize=(12, 4))
    if y_target is not None:
        ax = plt.subplot(2, 1, 1)
        dsp.waveplot(y_target, sr=sr)
        ax.set_title('Target waveform')
        ax = plt.subplot(2, 1, 2)
        dsp.waveplot(y_hat, sr=sr)
        ax.set_title('Prediction waveform')
    else:
        ax = plt.subplot(1, 1, 1)
        dsp.waveplot(y_hat, sr=sr)
        ax.set_title('Generated waveform')

    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()
