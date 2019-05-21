################################################################################
#
# Create a reproducible train/val/test split of the maestro data
#
# Author(s): Nik Vaessen
################################################################################

import sys
import os
import math

import numpy as np

from pianition import data_util

################################################################################
# Constants

split_fraction = (70, 20, 10)
minimum_num_samples_by_composer = 30

json_data_train_path = "train_path"
json_data_val_path = "val_path"
json_data_test_path = "test_path"
json_data_num_composers = "num_composers"
json_data_split_fraction = "split_fraction"


################################################################################
# Helper methods


def get_allowed_tracks():
    return data_util.get_allowed_paths(minimum_num_samples_by_composer)


def tracks_by_composer_id():
    allowed_tracks = get_allowed_tracks()

    track_by_composer = {}

    for track, ID in allowed_tracks:
        if ID not in track_by_composer:
            track_by_composer[ID] = []

        track_by_composer[ID].append(track)

    return track_by_composer


def create_split_paths():
    track_by_composer = tracks_by_composer_id()

    if sum(split_fraction) != 100:
        raise ValueError("split needs to sum up to 100")

    split = [s / 100 for s in split_fraction]

    train = []
    val = []
    test = []

    for ID, tracks in track_by_composer.items():
        n = len(tracks)

        num_val = math.ceil(n * split[1])
        num_test = math.ceil(n * split[2])

        num_train = n - num_val - num_test

        track_train = tracks[:num_train]
        tracks_val = tracks[num_train: num_train + num_val]
        tracks_test = tracks[-num_test:]

        train += track_train
        val += tracks_val
        test += tracks_test

    return train, val, test


################################################################################
# main function executing the splitting logic

def save_objects(output_path, object_list):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for idx, (id, sample) in enumerate(object_list):
        full_path = os.path.join(output_path, "sample{}.npz".format(idx))

        np.savez_compressed(full_path, (id, sample))


def main():
    if len(sys.argv) != 2:
        print("usage: python3 data_split path_to_storage")
        exit()

    root_path = sys.argv[1]
    print("saving data to", root_path)

    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    tr_paths, v_paths, t_paths = create_split_paths()

    # Training data
    print("extracting training data...")
    tr = data_util._get_data(tr_paths, split_data=False, only_first_window=True,
                             progress_bar=True)
    output_path = os.path.join(root_path, "train")
    save_objects(output_path, tr)

    tr = None
    print("training data saved to", output_path)

    # Validation data
    print("extracting validation data...")
    v = data_util._get_data(v_paths, split_data=False, only_first_window=True,
                            progress_bar=True)
    output_path = os.path.join(root_path, "val")
    save_objects(output_path, v)

    v = None
    print("validation data saved to", output_path)

    # Testing data
    print("extracting test data...")
    t = data_util._get_data(t_paths, split_data=False, only_first_window=True,
                            progress_bar=True)

    output_path = os.path.join(root_path, "test")
    save_objects(output_path, t)

    t = None
    print("test data saved to", output_path)


if __name__ == '__main__':
    main()
