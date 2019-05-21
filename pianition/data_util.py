################################################################################
# Utility methods for accessing and manipulating the (maestro) data (set).
#
# Author(s): Nik Vaessen
################################################################################

import os

import librosa
import numpy as np

from typing import Tuple, List

################################################################################
# Constants

root_dir = "data"

info_file_path = os.path.join(root_dir, "info.npz")

each_column_is_mfcc = True
time_axis = 1 if each_column_is_mfcc else 0
mffc_axis = 1 if time_axis == 0 else 0

json_mffc_hop_length = 'hop_length'
json_mffc_n_fft = 'n_fft'
json_mffc_id_to_composer = 'id_to_composer'
json_mffc_composer_to_id = 'composer_to_id'
json_mffc_song_to_id = 'song_to_id'
json_mffc_id_to_song = 'id_to_song'
json_mffc_n_samples = 'n_samples'
json_mffc_paths = 'paths'
json_mffc_paths_composer_id = 'paths_composer_id'
json_mffc_paths_song_id = 'paths_song_id'
# json_mfcc_paths_song_id

################################################################################
# Spectogram creation


def create_spectogram(audio_file_path, n_fft=2048, hop_length=1024):
    y, sr = librosa.load(audio_file_path)

    spect = librosa.feature.melspectrogram(y=y, sr=sr,
                                           n_fft=n_fft, hop_length=hop_length)
    spect = librosa.power_to_db(spect, ref=np.max)

    return spect.T

################################################################################
#


def _get_info() -> dict:
    info = np.load(info_file_path, allow_pickle=True)['info']

    return info.item()


def _load_sample(path: str) -> Tuple[int, np.ndarray]:
    sample = np.load(path, allow_pickle=True)['arr_0']

    if each_column_is_mfcc:
        sample[1] = sample[1].transpose()

    return sample


def _split_sample(sample: np.ndarray, window_size=256) -> List[np.ndarray]:
    num_splits = sample.shape[time_axis] // window_size

    if each_column_is_mfcc:
        sample = sample[:, 0:num_splits * window_size]
    else:
        sample = sample[0:num_splits * window_size, :]

    samples = np.split(sample, num_splits, axis=time_axis)

    return samples


def _extract_first_window(sample, window_size=256):
    if each_column_is_mfcc:
        return sample[:, 0:window_size]
    else:
        return sample[0:window_size, :]


def _get_data(paths,
              split_data=True,
              only_first_window=False,
              window_size=256,
              progress_bar=False):
    if split_data and only_first_window:
        raise ValueError("cannot split data AND only use first sample!")
    if not split_data and not only_first_window:
        raise ValueError("specify one of 'split_data' or 'use_first_window'")
    if window_size < 1 or not isinstance(window_size, int):
        raise ValueError("window size {} is invalid, "
                         "should be > 1 and integer".format(window_size))

    gave_warning = False
    data_tuples = []

    for idx, path in enumerate(paths):
        if progress_bar:
            print("\r{}/{}".format(idx, len(paths)), flush=True, end="")

        if os.path.exists(path):
            id, sample = _load_sample(path)

            if split_data:
                samples = [(id, sample) for sample in
                           _split_sample(sample, window_size=window_size)]
            else:
                samples = [(id, _extract_first_window(sample,
                                                      window_size=window_size))
                           ]

            data_tuples += samples

        elif not gave_warning:
            gave_warning = True
            print("Warning: one (or more) data paths are missing! (could "
                  "not find {})".format(path))

    if progress_bar:
        print()

    return data_tuples


def get_id_count():
    info = _get_info()

    count = {}
    path_id = info[json_info_paths_id]

    for id in path_id:
        if id in count:
            count[id] += 1
        else:
            count[id] = 1

    return count


def get_allowed_paths(use_only_count_bigger_than=30):
    info = _get_info()

    allowed_composers = [id for id, count in get_id_count().items()
                         if count > use_only_count_bigger_than]

    allowed_paths = []
    paths = info[json_info_paths]
    paths_id = info[json_info_paths_id]

    for path, id in zip(paths, paths_id):
        if id in allowed_composers:
            allowed_paths.append((path, id))

    return allowed_paths


def get_training_data():
    info = _get_info()
    paths = info[json_info_paths]


def get_validation_data():
    pass


def get_testing_data():
    pass


################################################################################

def main():
    dir = "/home/nik/kth/y1p1/speech/project/data/debug/"
    train_dir = os.path.join(dir, "train")
    val_dir = os.path.join(dir, "val")
    test_dir = os.path.join(dir, "test")

    for file in os.listdir(train_dir):
        obj = np.load(os.path.join(train_dir, file), allow_pickle=True)['arr_0']

        ID, sample = obj
        print(ID)
        # print(sample)
        print(sample.shape)
        print()



if __name__ == '__main__':
    main()
