import os
import numpy as np

import tensorflow as tf

from tacotron.utils.text import text_to_sequence
from hparams import hparams



_batches_per_group = 10
# pad input sequences with the <pad_token> 0 ( _ )
_pad = 0
# explicitly setting the padding to a value that doesn't originally exist in the spectrogram
# to avoid any possible conflicts, without affecting the output range of the model too much
if hparams.symmetric_mels:
    _target_pad = -(hparams.max_abs_value + .1)
else:
    _target_pad = -0.1
# Mark finished sequences with 1s
_token_pad = 1.


def _prepare_target(target, alignment):
    length = len(target)
    return _pad_target(target, _round_up(length, alignment))


def _prepare_token_targets(target, alignment):
    length = len(target) + 1
    return _pad_token_target(target, _round_up(length, alignment))


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_target_pad)


def _pad_token_target(t, length):
    return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=_token_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


class Feeder(object):

    def __init__(self, metadata_filename, shuffle):
        super(Feeder, self).__init__()

        self._hparams = hparams
        self.shuffle = shuffle
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

        self._mel_dir = os.path.join(os.path.dirname(metadata_filename), 'mels')
        self._linear_dir = os.path.join(os.path.dirname(metadata_filename), 'linear')

        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            frame_shift_ms = hparams.hop_size / hparams.sample_rate
            hours = sum([int(x[4]) for x in self._metadata]) * frame_shift_ms / 3600
            print('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

    def get_one_example(self):
        for meta in self._metadata:
            text = meta[5]

            input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
            input_length = input_data.shape[0]

            mel_target = np.load(os.path.join(self._mel_dir, meta[1]))
            # Create parallel sequences containing zeros to represent a non finished sequence
            token_target = np.asarray([0.] * (len(mel_target) - 1))
            linear_target = np.load(os.path.join(self._linear_dir, meta[2]))

            target_length = len(mel_target)

            # for decode, we need to make sure len(targets) % r == 0
            r = self._hparams.outputs_per_step
            mel_target = _prepare_target(mel_target, r)
            linear_target = _prepare_target(linear_target, r)
            token_target = _prepare_token_targets(token_target, r)

            yield input_data, mel_target, token_target, linear_target, input_length, target_length


def get_dataset(meta_file, shuffle):
    dataset = tf.data.Dataset.from_generator(generator=Feeder(meta_file, shuffle).get_one_example,
                                             output_types=(tf.int32, tf.float32, tf.float32, tf.float32, tf.int32,
                                                           tf.int32),
                                             output_shapes=(
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([None, hparams.num_mels]),
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([None, hparams.num_freq]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([]),
                                             ))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_batches_per_group*hparams.tacotron_batch_size)

    dataset = dataset.padded_batch(hparams.tacotron_batch_size,
                                   padded_shapes=([None], [None, None], [None], [None, None], [], []),
                                   padding_values=(_pad, _target_pad, _token_pad, _target_pad, _pad, _pad))
    # dataset = dataset.cache()
    return dataset.prefetch(buffer_size=4)


