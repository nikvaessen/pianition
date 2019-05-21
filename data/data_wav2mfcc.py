################################################################################
#
# Parse maestro data and converts wave files into mfcc features, along the way
# storing it into binary numpy files (.npz) format
#
# Simply run `python3 parse_data` in the same folder as 'maestro-v2.0.0.zip'
# and 'maestro-v2.0.0.json`
#
# Author(s): Nik Vaessen, Peter Mastnak
#
################################################################################

import os
import json

import numpy as np

from pianition import data_util

################################################################################
# Constants

data_dir = "mfcc"
sample_dir = os.path.join(data_dir, "samples")

meta_data_fn = os.path.join('maestro-v2.0.0.json')
data_zip_fn = os.path.join('maestro-v2.0.0.zip')
unzipped_dir_name = os.path.join('maestro-v2.0.0')

json_key_composer_name = 'canonical_composer'
json_key_song_name = 'canonical_title'
json_key_audio_file_path = 'audio_filename'

################################################################################
# Loading wave files and creation of spectrograms


def get_audio_path(audio_fn):
    return os.path.join(unzipped_dir_name, audio_fn)


def convert(audio_file, idx, total, n_fft=2048, hop_length=1024):
    print("\r{}/{}".format(idx, total), end='', flush=True)

    spectogram = data_util.create_spectogram(audio_file, n_fft=n_fft, hop_length=hop_length)

    return spectogram


def main():
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    with open(meta_data_fn, 'r') as f:
        d = json.load(f)

    unique_composers = set()
    unique_songs = set()
    data_samples = []

    for obj in d:
        composer_name = obj[json_key_composer_name]
        song_name = obj[json_key_song_name]
        audio_file = obj[json_key_audio_file_path]

        data_samples.append((composer_name, song_name, get_audio_path(audio_file)))
        unique_composers.add(composer_name)
        unique_songs.add(song_name)

    unique_composers = sorted(unique_composers)

    composer_to_id = {name: idx for idx, name in enumerate(unique_composers)}
    id_to_composer = {idx: name for name, idx in composer_to_id.items()}

    song_to_id = {name: idx for idx, name in enumerate(unique_songs)}
    id_to_song = {idx: name for name, idx in song_to_id.items()}

    n_fft = 2048
    hop_length = 1024

    paths = []
    paths_composer_id = []
    paths_song_id = []

    for idx, sample in enumerate(data_samples):
        composer_name, song_name, audio_file = sample
        composer_id = composer_to_id[composer_name]
        song_id = song_to_id[song_name]

        sample_path = os.path.join(sample_dir, "sample_{}.npz".format(idx))

        mfcc = convert(audio_file, idx, len(data_samples))
        saved_sample = (composer_id, mfcc)
        np.savez_compressed(sample_path, saved_sample)

        paths.append(sample_path)
        paths_composer_id.append(composer_id)
        paths_song_id.append(song_id)

    n_samples = len(data_samples)

    info = {
        data_util.json_mffc_n_samples: n_samples,
        data_util.json_mffc_composer_to_id: composer_to_id,
        data_util.json_mffc_id_to_composer: id_to_composer,
        data_util.json_mffc_song_to_id: song_to_id,
        data_util.json_mffc_id_to_song: id_to_song,
        data_util.json_mffc_n_fft: n_fft,
        data_util.json_mffc_hop_length: hop_length,
        data_util.json_mffc_paths: paths,
        data_util.json_mffc_paths_composer_id: paths_composer_id,
        data_util.json_mffc_paths_song_id: paths_song_id
    }

    with open(os.path.join(data_dir, "mfcc.json"), 'w') as f:
        json.dump(info, f)


if __name__ == '__main__':
    main()
