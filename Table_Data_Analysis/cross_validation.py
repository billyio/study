import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PATH = './'
df = pd.read_csv(os.path.join(PATH, 'application_train.csv'), nrows=None)
X = df[[col for col in df.columns if col not in ['TARGET']]]
y = df.TARGET

folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=2019)

#分割したデータのindexを返す
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print('{0}fold'.format(n_fold))
    print('TRAIN', train_idx)
    print('VALID', valid_idx)
    print(X.iloc[train_idx].shape) #dfはindex指定で読み込むときilocを使う


import numpy as np
from sklearn.model_selection import KFold
x = np.array([[1, 2], [3, 4], [4, 3], [1, 2], [3, 4], [5, 6], [6, 4], [1, 2], [2, 2], [1, 2]])
y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 1])
kf = KFold(n_splits = 5, shuffle = True)

for train_index, test_index in kf.split(x):
    print(train_index, test_index)