from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np

import sys
import os
import gc

sys.path.insert(0, '..')
from pianition.data_util import load_dataset

root_path1 = '/media/drive/data/'
root_path2 = '../data/'

if os.path.exists(root_path1):
    root_path = root_path1
elif os.path.exists(root_path2):
    root_path = root_path2
else:
    print('could not find root path')
    exit()

print('using root_path', root_path)

paths = [
    # 'debug128', 'debug256', 'debug512', 'debug768',
    'full128', 'full256', 'full512', 'full768'
]

result_txt_path = 'result_knn.txt'


def test(use_min_max_scaler=False):
    for path in paths:
        ds = load_dataset(os.path.join(root_path, path))

        with open(result_txt_path, 'a') as f:
            f.write("{}\n".format(path))

        mfcc, label = ds.get_train_full(flatten=True, output_encoded=False)
        mfcc_val, label_val = ds.get_val_full(flatten=True, output_encoded=False)
        mfcc_test, label_test = ds.get_test_full(flatten=True, output_encoded=False)

        if use_min_max_scaler:
            mfcc = MinMaxScaler(copy=False).fit_transform(mfcc)
            mfcc_val = MinMaxScaler(copy=False).fit_transform(mfcc_val)
            mfcc_test = MinMaxScaler(copy=False).fit_transform(mfcc_test)

        print('training')
        print(mfcc.shape, np.max(mfcc), np.min(mfcc))
        print(label.shape)

        print('val')
        print(mfcc_val.shape, np.max(mfcc_val), np.min(mfcc_val))
        print(label_val.shape)

        print('test')
        print(mfcc_test.shape, np.max(mfcc_test), np.min(mfcc_test))
        print(label_test.shape)

        print('')

        best_knn = None
        best_score = float('-inf')

        for k in range(1, 9, 2):
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-2)
            knn.fit(mfcc, label)
            score = knn.score(mfcc_val, label_val)

            with open(result_txt_path, 'a') as f:
                f.write('k={} --> {}'.format(k, score))

            if score > best_score:
                best_knn = knn
                best_score = score

            knn = None
            gc.collect()

        print('accuracy on test:')
        score_val = 'look above'
        score_test = best_knn.score(mfcc_test, label_test)
        print('test accuracy:', score_test)

        with open(result_txt_path, 'a') as f:
            f.write("{}\n".format(path))
            f.write("val acc: {}\ntest acc:{}\n".format(score_val, score_test))

        mfcc, label = None, None
        mfcc_val, label_val = None, None
        mfcc_test, label_test = None, None

        gc.collect()

        print(end="\n\n\n\n\n\n\n")


test(use_min_max_scaler=True)

