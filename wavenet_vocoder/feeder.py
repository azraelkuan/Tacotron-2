import os
import numpy as np
import tensorflow as tf

from datasets import audio
from hparams import hparams


class Feeder(object):

    def __init__(self, metadata_filename, hparams):

        if hparams.gin_channels > 0:
            raise NotImplementedError(
                'Global conditioning preprocessing has not been added yet, '
                'it will be out soon. Thanks for your patience!')
        self._hparams = hparams

        self._data_dir = os.path.dirname(metadata_filename)
        self._mel_dir = os.path.join(self._data_dir, 'mels')
        self._audio_dir = os.path.join(self._data_dir, 'audio')

        with open(metadata_filename, 'r') as f:
            self._metadata = [line.strip().split('|') for line in f]
            frame_shift_ms = hparams.hop_size / hparams.sample_rate
            hours = sum([int(x[4]) for x in self._metadata]) * frame_shift_ms / 3600
            print('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

        self.local_condition, self.global_condition = self._check_conditions()

    def _check_conditions(self):
        local_condition = self._hparams.cin_channels > 0
        global_condition = self._hparams.gin_channels > 0
        return local_condition, global_condition

    def _limit_time(self):
        """Limit time resolution to save GPU memory."""
        if self._hparams.max_time_sec is not None:
            return int(self._hparams.max_time_sec * self._hparams.sample_rate)
        elif self._hparams.max_time_steps is not None:
            return self._hparams.max_time_steps
        else:
            return None

    def _assert_ready_for_upsample(self, x, c):
        assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size(self._hparams)

    def _adjust_time_step(self, audio_data, local_feature, max_time_steps):
        """Adjust time resolution for local condition."""
        hop_size = audio.get_hop_size(self._hparams)
        if local_feature is not None:
            if self._hparams.upsample_conditional_features:
                self._assert_ready_for_upsample(audio_data, local_feature)
                if max_time_steps is not None:
                    max_steps = _ensure_divisible(max_time_steps, hop_size, True)
                    if len(audio_data) > max_time_steps:
                        max_time_frames = max_steps // hop_size
                        start = np.random.randint(0, len(local_feature) - max_time_frames)
                        time_start = start * hop_size
                        audio_data = audio_data[time_start:time_start + max_time_frames * hop_size]
                        local_feature = local_feature[start:start + max_time_frames, :]
                        self._assert_ready_for_upsample(audio_data, local_feature)
            else:
                audio_data, local_feature = audio.adjust_time_resolution(audio_data, local_feature, self._hparams)
                if max_time_steps is not None and len(audio_data) > max_time_steps:
                    s = np.random.randint(0, len(audio_data) - max_time_steps)
                    audio_data, local_feature = audio_data[s:s + max_time_steps], local_feature[s:s + max_time_steps, :]
                assert len(audio_data) == len(local_feature)

        return audio_data, local_feature

    def get_one_example(self):
        for meta in self._metadata:
            audio_file = meta[0]
            input_data = np.load(os.path.join(self._audio_dir, audio_file))
            if self.local_condition:
                mel_file = meta[1]
                local_feature = np.load(os.path.join(self._mel_dir, mel_file))
            else:
                local_feature = False

            # ===== To Do ===== #
            global_feature = False

            # adjust time step for local condition
            max_time_step = self._limit_time()
            input_data, local_feature = self._adjust_time_step(input_data, local_feature, max_time_step)
            target_data = input_data
            input_length = len(input_data)

            yield input_data, target_data, input_length, local_feature, global_feature


def _ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def get_dataset(meta_file, shuffle):
    feeder = Feeder(meta_file, hparams)

    if feeder.local_condition and feeder.global_condition:
        output_types = (tf.float32, tf.float32, tf.int32, tf.float32, tf.int32)
        output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]),
                         tf.TensorShape([None, hparams.num_mels]), tf.TensorShape([]))
    elif feeder.local_condition:
        output_types = (tf.float32, tf.float32, tf.int32, tf.float32, tf.bool)
        output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]),
                         tf.TensorShape([None, hparams.num_mels]), tf.TensorShape([]))
    elif feeder.global_condition:
        output_types = (tf.float32, tf.float32, tf.int32, tf.bool, tf.int32)
        output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]),
                         tf.TensorShape([]))
    else:
        output_types = (tf.float32, tf.float32, tf.int32, tf.bool, tf.bool)
        output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]),
                         tf.TensorShape([]))

    dataset = tf.data.Dataset.from_generator(generator=feeder.get_one_example,
                                             output_types=output_types,
                                             output_shapes=output_shapes)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(hparams.wavenet_batch_size)
    return dataset.prefetch(hparams.wavenet_batch_size * 4)


if __name__ == '__main__':

    dataset = get_dataset('/mnt/lustre/sjtu/users/kc430/data/my/tacotron2/ljspeech/train.txt', shuffle=True)
    iterator = dataset.make_one_shot_iterator()
    next = iterator.get_next()

    with tf.Session() as sess:
        print(sess.run(next))
