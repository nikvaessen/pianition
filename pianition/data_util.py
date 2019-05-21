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

each_column_is_mfcc = True
time_axis = 1 if each_column_is_mfcc else 0
mffc_axis = 1 if time_axis == 0 else 0



################################################################################
# Spectogram creation


################################################################################
#

def load_dataset(path: str):
    print("loading dataset from", path, "...")


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
