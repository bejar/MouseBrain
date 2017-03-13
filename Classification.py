"""
.. module:: Classification

Classification
*************

:Description: Classification

    

:Authors: bejar
    

:Version: 

:Created on: 08/03/2017 10:28 

"""

from MouseBrain.Config import data_path
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

__author__ = 'bejar'

if __name__ == '__main__':
    X = np.load(data_path + 'mousepre.npy')
    y = np.load(data_path + 'mouselabels.npy')

    X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.2, stratify=y)

    cnt = Counter(y_train)
    print np.array([cnt[0], cnt[1]])/float(y_train.shape[0])

    clf = SVC(C=10, kernel='poly', degree=2)

    cvvals = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=10), n_jobs=-1)
    sc = np.mean(cvvals)

    print sc

    clf.fit(X_train, y_train)

    print classification_report(y_test, clf.predict(X_test))
    print confusion_matrix(y_test, clf.predict(X_test))



