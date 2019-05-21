################################################################################
#
# Create a reproducible train/val/test split of the maestro data
#
# Author(s): Nik Vaessen
################################################################################

import math

from pianition import data_util

################################################################################
# Constants

split_fraction = (70, 20, 10)
print("split", split_fraction)
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
        tracks_val = tracks[num_train: num_train+num_val]
        tracks_test = tracks[-num_test:]

        train += track_train
        val += tracks_val
        test += tracks_test

    return train, val, test


################################################################################
# main function executing the splitting logic


def main():
    tr_paths, v_paths, t_paths = create_split_paths()

    print(len(tr_paths), tr_paths[0:2])
    print(len(v_paths), v_paths[0:2])
    print(len(t_paths), t_paths[0:2])

    tr = data_util._get_data(tr_paths, split_data=False, only_first_window=True)
    # v = data_util._get_data(v_paths, split_data=False, only_first_window=True)
    # t = data_util._get_data(t_paths, split_data=False, only_first_window=True)

    print(len(tr), tr[0].shape)
    # print(len(v), v[0].shape)
    # print(len(t), t[0].shape)


if __name__ == '__main__':
    main()
