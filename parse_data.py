################################################################################
#
# Parse maestro data and converts it into binary numpy files (.npz) format
# usable for training purposes
#
# Simply run `python3 parse_data` in the same folder as 'maestro-v2.0.0.zip'
# and 'maestro-v2.0.0.json`
#
# Author(s): Nik Vaessen, Peter Mastnak
#
################################################################################

import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display as display

################################################################################
# Constants

meta_data_fn = 'maestro-v2.0.0.json'
data_zip_fn = 'maestro-v2.0.0.zip'
unzipped_dir_name = 'maestro-v2.0.0'

json_key_composer_name = 'canonical_composer'
json_key_audio_file_path = 'audio_filename'


################################################################################
# Loading wave files and creation of spectrograms


def get_audio_path(audio_fn):
    return os.path.join(unzipped_dir_name, audio_fn)


def create_spectogram(audio_file_path):
    y, sr = librosa.load(audio_file_path)

    spect = librosa.feature.melspectrogram(y=y, sr=sr,
                                           n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)

    return spect.T


################################################################################
# Main function executing parsing logic

def convert(sample, composer_to_id):
    name, file_path = sample

    id = composer_to_id[name]
    spectogram = create_spectogram(file_path)

    return id, spectogram


def main():
    with open(meta_data_fn, 'r') as f:
        d = json.load(f)

    unique_composers = set()
    data_samples = []

    for idx, obj in enumerate(d):
        if idx % 10 == 0:
            print("\r{}/{}".format(idx, len(d)), end='', flush=True)

        name = obj[json_key_composer_name]
        audio_file = obj[json_key_audio_file_path]

        data_samples.append((name, get_audio_path(audio_file)))
        unique_composers.add(name)
        break

    unique_composers = sorted(unique_composers)

    composer_to_id = {name: idx for idx, name in enumerate(unique_composers)}
    id_to_composer = {idx: name for idx, name in composer_to_id.items()}

    data_samples = [convert(sample, composer_to_id) for sample in data_samples]

    print(data_samples[0])

    pass


if __name__ == '__main__':
    main()
