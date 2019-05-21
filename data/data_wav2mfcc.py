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

data_dir = "."
sample_dir = os.path.join(data_dir, "samples")

meta_data_fn = 'maestro-v2.0.0.json'
data_zip_fn = 'maestro-v2.0.0.zip'
unzipped_dir_name = 'maestro-v2.0.0'

json_key_composer_name = 'canonical_composer'
json_key_audio_file_path = 'audio_filename'


################################################################################
# Loading wave files and creation of spectrograms


def get_audio_path(audio_fn):
    return os.path.join(unzipped_dir_name, audio_fn)


def convert(sample, composer_to_id, idx, n_fft=2048, hop_length=1024):
    print("\r{}/{}".format(idx[0], idx[1]), end='', flush=True)

    name, file_path = sample

    id = composer_to_id[name]
    spectogram = data_util.create_spectogram(file_path, n_fft=n_fft, hop_length=hop_length)

    return id, spectogram


def main():
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    with open(meta_data_fn, 'r') as f:
        d = json.load(f)

    unique_composers = set()
    data_samples = []

    for obj in d:
        name = obj[json_key_composer_name]
        audio_file = obj[json_key_audio_file_path]

        data_samples.append((name, get_audio_path(audio_file)))
        unique_composers.add(name)

    unique_composers = sorted(unique_composers)

    composer_to_id = {name: idx for idx, name in enumerate(unique_composers)}
    id_to_composer = {idx: name for name, idx in composer_to_id.items()}

    n_fft = 2048
    hop_length = 1024

    paths = []
    paths_id = []
    for idx, sample in enumerate(data_samples):
        sample = convert(sample, composer_to_id, (idx, len(data_samples)))
        sample_path = os.path.join(sample_dir, "sample_{}.npz".format(idx))

        np.savez_compressed(sample_path, sample)

        paths.append(sample_path)
        paths_id.append(sample[0])

    n_samples = len(data_samples)

    info = {
        data_util.json_info_n_samples: n_samples,
        data_util.json_info_composer_to_id: composer_to_id,
        data_util.json_info_id_to_composer: id_to_composer,
        data_util.json_info_n_fft: n_fft,
        data_util.json_info_hop_length: hop_length,
        data_util.json_info_paths: paths,
        data_util.json_info_paths_id: paths_id
    }

    np.savez_compressed(os.path.join(data_dir, "info.npz"), info=info)

    pass


if __name__ == '__main__':
    main()
