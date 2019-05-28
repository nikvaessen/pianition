from sklearn.neighbors import KNeighborsClassifier

import sys

sys.path.insert(0, '..')

from pianition.data_util import load_dataset

ds = load_dataset('../data/full/')

mfcc, label = ds.get_train_full(flatten=True)
mfcc_val, label_val = ds.get_val_full(flatten=True)
mfcc_test, label_test = ds.get_test_full(flatten=True)

print('training')
print(mfcc.shape)
print(label.shape)

print('val')
print(mfcc_val.shape)
print(label_val.shape)

print('test')
print(mfcc_test.shape)
print(label_test.shape)

# In[ ]:


best_knn = None
best_score = float('-inf')

for k in range(1, 19, 2):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=1)
    knn.fit(mfcc, label)
    score = knn.score(mfcc_val, label_val)
    print('k={} --> {}'.format(k, score))

    if score > best_score:
        best_knn = knn
        best_score = score

    knn = None

print('accuracy on test:')
score = best_knn.score(mfcc_test, label_test)
print(score)




