"""
.. module:: SignalIntegral

SignalIntegral
*************

:Description: SignalIntegral

    

:Authors: bejar
    

:Version: 

:Created on: 26/04/2017 13:22 

"""

from MouseBrain.Config import data_path
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
from matplotlib.patches import Rectangle
from scipy.signal import argrelextrema, argrelmax
from MouseBrain.Util.Misc import reduceseg

__author__ = 'bejar'

def max_integral(signal, wlen):
    """
    Return the integral and the position of the window with maximal integral
    integral = sum of the absilute value of the signal  
     
    :param wlen: 
    :return: 
    """

    smax = 0
    pos = 0
    lint = []
    for j in range(signal.shape[0] - wlen):
        sact = np.sum(np.abs(signal[j:j + wlen]))
        lint.append(sact)
        if sact > smax:
            smax = sact
            pos = j

    return smax, pos, lint


if __name__ == '__main__':
    X = np.load(data_path + 'mousepre1.npy')
    Y = np.load(data_path + 'mousepost1.npy')
    id = np.load(data_path + 'mouseids1.npy')

    wlen = 25
    for s in range(X.shape[0]):
        vmx, pmx, lint = max_integral(X[s], wlen)
        lint = np.array(lint)
        fig = plt.figure(figsize=(10, 40))
        ax = fig.add_subplot(211)
        ax.plot(range(X.shape[1]), X[s], 'r')
        smax = reduceseg(argrelextrema(X[s], np.greater_equal, order=wlen)[0], 1)
        vsel = np.zeros(X.shape[1])
        for i in smax:
            vsel[i] = X[s,i]
        ax.plot(range(X.shape[1]), vsel, 'b')
        ax.add_patch(Rectangle((pmx,np.min(X[s])), wlen, np.max(X[s])-np.min(X[s]), facecolor="grey"))

        smax = reduceseg(argrelextrema(lint, np.greater_equal, order=3)[0], 1)
        vsel = np.zeros(X.shape[1])
        for i in smax:
            vsel[i] = lint[i]

        mxvals = lint[smax]
        print np.std(mxvals)
        ax = fig.add_subplot(212)
        ax.plot(range(X.shape[1]), vsel, 'b')
        ax.plot(range(len(lint)), lint, 'r')
        # ax.plot(range(len(lint)-2), np.diff(lint, 2), 'r')
        plt.show()
