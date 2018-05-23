#!/bin/bash

#SBATCH -p fatq
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -J pre
#SBATCH -o log/pre.log


source activate dl

# for ljspeech
python preprocess.py \
    --base_dir "/mnt/lustre/sjtu/users/kc430/data/sjtu/tts/text-to-speech/english" \
    --dataset "LJSpeech-1.0" \
    --output "/mnt/lustre/sjtu/users/kc430/data/my/tacotron2/ljspeech" \
    --n_jobs 10