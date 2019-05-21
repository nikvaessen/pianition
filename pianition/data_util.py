################################################################################
# Utility methods for accessing and manipulating the (maestro) data (set).
#
# Author(s): Nik Vaessen
################################################################################

import os
import pathlib
import json

import numpy as np

from keras.utils import to_categorical

################################################################################
# Constants

json_data_train = "train"
json_data_val = "val"
json_data_test = "test"

json_data_set_path = 'path'
json_data_set_composer_label = 'composer-label'
json_data_set_song_label = 'song-label'

json_data_used_composer_id = "num_composers"
json_data_used_song_id = "split_fraction"

json_data_label_to_composer = 'label_to_composer'
json_data_label_to_song = 'label_to_song'


################################################################################

def get_abs_paths(root, paths):
    corrected_paths = []

    for path in paths:
        corrected_paths.append(
            os.path.join(root, *pathlib.Path(path).parts[1:])
        )

    return corrected_paths


def load_dataset(path: str):
    print("loaded dataset from {}".format(path))

    info_object = os.path.join(path, "info.json")

    if not os.path.isfile(info_object):
        raise ValueError("could not find 'info.json' in {}".format(
            path
        ))

    with open(info_object, 'r') as f:
        info = json.load(f)

    train = info[json_data_train]
    train_paths = get_abs_paths(path, train[json_data_set_path])
    train_labels = train[json_data_set_composer_label]

    val = info[json_data_val]
    val_paths = get_abs_paths(path, val[json_data_set_path])
    val_labels = val[json_data_set_composer_label]

    test = info[json_data_test]
    test_paths = get_abs_paths(path, test[json_data_set_path])
    test_labels = test[json_data_set_composer_label]

    num_classes = len(info[json_data_label_to_composer].keys())

    return Dataset(
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        num_classes
    )


def load_paths(paths):
    return [np.load(path, allow_pickle=True)['arr_0'] for path in paths]


class Dataset:

    def __init__(self,
                 train_paths,
                 train_labels,
                 val_paths,
                 val_labels,
                 test_paths,
                 test_labels,
                 num_classes
                 ):
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.val_paths = val_paths
        self.val_labels = val_labels
        self.test_paths = test_paths
        self.test_labels = test_labels

        self.num_classes = num_classes

    def get_train_full(self):
        x = load_paths(self.train_paths)
        y = [to_categorical(label, num_classes=self.num_classes)
             for label in self.train_labels]

        return x, y

    def get_val_full(self):
        x = load_paths(self.val_paths)
        y = [to_categorical(label, num_classes=self.num_classes)
             for label in self.val_labels]

        return x, y

    def get_test_full(self):
        x = load_paths(self.test_paths)
        y = [to_categorical(label, num_classes=self.num_classes)
             for label in self.test_labels]

        return x, y


################################################################################


def quick_test():
    dir = "/home/nik/kth/y1p1/speech/project/data/debug/"
    train_dir = os.path.join(dir, "train")
    val_dir = os.path.join(dir, "val")
    test_dir = os.path.join(dir, "test")

    for file in os.listdir(train_dir):
        obj = np.load(os.path.join(train_dir, file), allow_pickle=True)['arr_0']

        mfcc = obj
        print(mfcc)
        print(mfcc.shape)
        print()
        break


def main():
    dir = "/home/nik/kth/y1p1/speech/project/data/debug/"

    ds = load_dataset(dir)

    x, y = ds.get_train_full()

    print(len(x))
    print(len(y))

    print(x[0], x[0].shape)
    print(y[0], y[0].shape)


if __name__ == '__main__':
    main()
