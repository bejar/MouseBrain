"""
.. module:: Analysis

Analysis
*************

:Description: Analysis

    

:Authors: bejar
    

:Version: 

:Created on: 28/03/2017 11:05 

"""

__author__ = 'bejar'


from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
from MouseBrain.Data import Dataset
from MouseBrain.Config import data_path
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn


def integral(pre, post):
    """
    Integral of the signals
    
    :param pre: 
    :param post: 
    :return: 
    """
    lintpre = [np.sum(np.abs(X[i])) for i in range(pre.shape[0])]
    lintpost = [np.sum(np.abs(Y[i])) for i in range(post.shape[0])]


    fig = plt.figure(figsize=(10,10))
    sn.distplot(lintpre)
    plt.show()
    fig = plt.figure(figsize=(10,10))
    sn.distplot(lintpost)
    plt.show()

    fig = plt.figure(figsize=(10,10))
    plt.scatter(lintpre, lintpost)
    plt.show()


def study(X,Y, ids, title, wlenpre, wlenpos):
    """
    Study of mouse events
    
    :param X: 
    :param Y: 
    :return: 
    """
    postmid = int(Y.shape[0])
    lintpre = [np.sum(np.abs(X[i])) for i in range(X.shape[0])]
    lintpost = [np.sum(np.abs(Y[i,0:postmid])) for i in range(Y.shape[0])]

    lpreint = []
    lprepos = []
    for i in range(X.shape[0]):
        smax = 0
        pos = 0
        for j in range(X.shape[1]-wlenpre):
            sact = np.sum(np.abs(X[i,j:j+wlenpre]))
            if sact > smax:
                smax = sact
                pos = j + (wlenpre/2)
        lpreint.append(smax)
        lprepos.append(pos)

    lpreint = np.array(lpreint)

    sel = lpreint > 30

    lposint = []
    lpospos = []
    for i in range(Y.shape[0]):
        smax = 0
        pos = 0
        for j in range(X.shape[1]-wlenpos):
            sact = np.sum(np.abs(Y[i,j:j+wlenpos]))
            if sact > smax:
                smax = sact
                pos = j + (wlenpos/2)
        lposint.append(smax)
        lpospos.append(pos)

    fig = plt.figure(figsize=(30,10))
    ax = fig.add_subplot(131)
    plt.scatter(lpreint, lintpost, c=sel)
    ax.set_xlabel('maxima integral ventana pre')
    ax.set_ylabel('Integral post')
    ax = fig.add_subplot(132)
    plt.scatter(lprepos, lintpost, c=sel)
    ax.set_xlabel('posicion maxima integral ventana pre')
    ax.set_ylabel('Integral post')
    ax = fig.add_subplot(133)
    plt.scatter(lprepos, lpreint, c=sel)
    ax.set_xlabel('posicion maxima integral ventana pre')
    ax.set_ylabel('maxima integral ventana pre')
    fig.suptitle(title + ' Wpr= '+str(wlenpre))
    plt.show()

    fig = plt.figure(figsize=(30,10))
    fig.suptitle(title + ' Wpr= '+str(wlenpre) + ' Wpo= '+str(wlenpos))
    ax = fig.add_subplot(131)
    ax.set_xlabel('maxima integral ventana pre')
    ax.set_ylabel('maxima integral ventana post')
    plt.scatter(lpreint, lposint, c=sel)
    ax = fig.add_subplot(132)
    ax.set_xlabel('posicion maxima integral ventana pre')
    ax.set_ylabel('maxima integral ventana post')
    plt.scatter(lprepos, lposint, c=sel)
    ax = fig.add_subplot(133)
    ax.set_xlabel('posicion maxima integral ventana pre')
    ax.set_ylabel('posicion maxima integral ventana post')
    plt.scatter(lprepos, lpospos, c=sel)
    plt.show()

    print ids[sel]
    #
    # fig = plt.figure(figsize=(20,10))
    # ax = fig.add_subplot(121)
    # plt.scatter(lpreint, lpospos)
    # ax = fig.add_subplot(122)
    # plt.scatter(lprepos, lpospos)
    #
    # plt.show()

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    # plt.scatter(lpreint, lprepos, zs=lposint, depthshade=False, c='r', s=100)
    # plt.show()



__author__ = 'bejar'

if __name__ == '__main__':
    # X = np.load(data_path + 'mousepre1.npy')
    # Y = np.load(data_path + 'mousepost1.npy')
    # id = np.load(data_path + 'mouseids1.npy')
    # study(X,Y, id, 'Evento positivo', 25, 25)
    X = np.load(data_path + 'mousepre0.npy')
    Y = np.load(data_path + 'mousepost0.npy')
    id = np.load(data_path + 'mouseids0.npy')
    study(X,Y, id, 'Evento Negativo', 25, 25)
