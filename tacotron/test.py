import os
import tensorflow as tf
from tacotron.synthesizer import Synthesizer

from hparams import hparams

output_dir = 'logs-tacotron2/eval'


def run_eval():
    checkpoint_path = tf.train.get_checkpoint_state("logs-tacotron2/checkpoints/").model_checkpoint_path
    wav_dir = os.path.join(output_dir, 'wavs')
    plot_dir = os.path.join(output_dir, 'plots')
    mel_dir = os.path.join(output_dir, 'mels')
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)

    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, gta=False, model_name='Tacotron')
    for i, text in enumerate(hparams.sentences):
        print(text)
        mel_filename = synth.synthesize(text, i + 1, mel_dir, output_dir, None)
        pass


if __name__ == '__main__':
    run_eval()
