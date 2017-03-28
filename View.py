"""
.. module:: View

View
*************

:Description: View

    

:Authors: bejar
    

:Version: 

:Created on: 13/03/2017 8:26 

"""

from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
from MouseBrain.Data import Dataset
from MouseBrain.Config import data_path
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'bejar'

if __name__ == '__main__':
    col = ['r', 'y']
    X = np.load(data_path + 'mousepre.npy')
    X = np.load(data_path + 'mousepost.npy')
    y = np.load(data_path + 'mouselabels.npy')

    # transf = MDS(n_components=2, n_jobs=-1, random_state=0)
    transf = SpectralEmbedding(n_components=3, affinity='rbf', gamma=0.001, n_neighbors=25)
    # transf = Isomap(n_components=2, n_neighbors=25, n_jobs=-1)

    fdata = transf.fit_transform(X)
    # print(transf.stress_)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    #
    plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100, c=[col[i] for i in y])
    # plt.scatter(fdata[:, 0], fdata[:, 1], c=[col[i] for i in y])

    plt.show()
