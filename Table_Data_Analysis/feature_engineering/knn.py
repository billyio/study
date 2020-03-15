knn

参照
・github, python実装
# https://github.com/upura/knnFeat

・u++の解説
# https://upura.hatenablog.com/entry/2018/06/23/165855#K%E8%BF%91%E5%82%8D%E3%82%92%E7%94%A8%E3%81%84%E3%81%9F%E7%89%B9%E5%BE%B4%E9%87%8F%E6%8A%BD%E5%87%BA

from knnFeat import knnExtract
newX = knnExtract(X, y, k = 1, folds = 5)

# 上のやつ
import numpy as np
from sklearn.model_selection import KFold

def _distance(a, b):
    return np.linalg.norm(b - a)

def _get_feat(data, X_train, y_train, class_index, k_index):
    inclass_X = X_train[y_train == class_index]
    distances = np.array([_distance(a, data) for a in inclass_X])
    sorted_distances_index = np.argsort(distances)
    nearest_index = list(sorted_distances_index[0: (k_index + 1)])
    dist = np.sum(distances[nearest_index])
    return dist

def knnExtract(X, y, k=1, folds=5):
    CLASS_NUM = len(set(y))
    res = np.empty((len(X), CLASS_NUM * k))
    kf = KFold(n_splits=folds,  shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]

        features = np.empty([0, len(X_test)])

        for class_index in range(CLASS_NUM):
            for k_index in range(k):
                feat = np.array([np.apply_along_axis(
                    _get_feat, 1,
                    X_test, X_train, y_train,
                    class_index, k_index
                )])
                features = np.append(features, feat, axis=0)
        res[test_index] = features.T

    return res
