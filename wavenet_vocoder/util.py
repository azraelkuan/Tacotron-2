import numpy as np


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


# From https://github.com/r9y9/wavenet_vocoder/blob/master/lrschedule.py
def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000):
    # Noam scheme from tensor2tensor:
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps ** 0.5 * np.minimum(
        step * warmup_steps ** -1.5, step ** -0.5)
    return lr


def step_learning_rate_decay(init_lr, global_step,
                             anneal_rate=0.98,
                             anneal_interval=30000):
    return init_lr * anneal_rate ** (global_step // anneal_interval)


def cyclic_cosine_annealing(init_lr, global_step, T, M):
    """Cyclic cosine annealing

    https://arxiv.org/pdf/1704.00109.pdf

    Args:
        init_lr (float): Initial learning rate
        global_step (int): Current iteration number
        T (int): Total iteration number (i,e. nepoch)
        M (int): Number of ensembles we want

    Returns:
        float: Annealed learning rate
    """
    TdivM = T // M
    return init_lr / 2.0 * (np.cos(np.pi * ((global_step - 1) % TdivM) / TdivM) + 1.0)
