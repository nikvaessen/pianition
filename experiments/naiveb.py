from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np

import sys
import os

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


paths = ['debug128', 'debug256', 'debug512', 'debug768',
         'full128', 'full256', 'full512', 'full768']


def test(use_min_max_scaler=False):
    for path in paths:
        ds = load_dataset(os.path.join(root_path, path))

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
        nb = MultinomialNB()
        nb.fit(mfcc, label)
        score_val = nb.score(mfcc_val, label_val)
        score_test = nb.score(mfcc_test, label_test)
        print('val accuracy:', score_val)
        print('test accuracy:', score_test)
        print(end="\n\n\n\n\n\n\n")


test(use_min_max_scaler=True)
