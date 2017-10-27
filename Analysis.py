"""
.. module:: Analysis

Analysis
*************

:Description: Analysis

    

:Authors: bejar
    

:Version: 

:Created on: 28/03/2017 11:05 

"""

from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
from MouseBrain.Data import Dataset
from MouseBrain.Config import data_path
from MouseBrain.Util.Misc import max_integral, reduceseg
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
from scipy.signal import argrelextrema
from pymongo import MongoClient
import cPickle
from scipy.stats import ks_2samp, anderson_ksamp

__author__ = 'bejar'


def accum_sep(vdata):
    """

    :param vdata:
    :return:
    """
    smax = argrelextrema(vdata, np.greater_equal, order=10)[0]
    dsel = vdata[smax]
    cnt = len(dsel)
    if cnt > 1:
        p = np.argmax(dsel)
        dsel[p] = -1
        end = False
        cnt -= 1
        while not end:
            p1 = np.argmax(dsel)
            cnt -= 1
            if np.abs(smax[p1] - smax[p]) < 50:
                dsel[p1] = -1
            else:
                end = True
            if cnt == 0:
                end = True
        return np.abs(smax[p1] - smax[p])
    else:
        return 0


def plot_positions(id, eleft, eright, axes, data, mxstd, tol=2, eclass=True, new=False):
    """
    Plots the data for the pre and post events
    :param eleft: 
    :param eright: 
    :param ax1: 
    :param ax2: 
    :param data: 
    :param mxstd: 
    :param tol: 
    :param eclass: 
    :return: 
    """
    print '------------ %d < %d' % (eleft, eright)
    ax1, ax2, ax3, ax4 = axes
    ax1.plot([0, 0], [0, 180], 'k')
    ax2.plot([0, 0], [0, 180], 'k')
    lprepos, lpreint, lpospos, lposint = data
    for ip, (prep, prei, posp, posi, idi) in enumerate(zip(lprepos, lpreint, lpospos, lposint, id)):
        mark = 's' if new and ((idi / 1000) < 100) else 'o'
        if eleft < prep < eright:
            if (eclass and prei > posi) or (not eclass and prei < posi):
                ax1.plot([prep, posp], [prei, posi], 'g:')

                clm = 'b' if mark == 'o' else 'c'
                cm = 'k' if mxstd[ip] > tol else clm
                ax1.plot([prep], [prei], cm, marker=mark)
                ax1.plot([posp], [posi], clm, marker=mark)
                print('+', id[ip], prei, posi)
                ax3.plot(posi, prei, clm, marker=mark)
                if prei >= posi:
                    ax3.set_xlabel('POST < PRE')
                else:
                    ax3.set_xlabel('POST > PRE')
            else:
                clm = 'y' if mark == 'o' else 'g'
                cm = 'k' if mxstd[ip] > tol else clm
                ax2.plot([prep], [prei], cm, marker=mark)
                ax2.plot([posp], [posi], clm, marker=mark)
                ax2.plot([prep, posp], [prei, posi], 'r:')
                ax4.plot(posi, prei, clm, marker=mark)
                print('-', id[ip], prei, posi)
                if prei >= posi:
                    ax4.set_xlabel('POST < PRE')
                else:
                    ax4.set_xlabel('POST > PRE')


def study2(X, Y, id, title, wlenpre, wlenpos, off=0, freq=0, eclass=True, tol=4, method='integral', new=False, testing=False):
    """
    Study of mouse events
    
    :param X: 
    :param Y: 
    :return: 
    """
    postmid = int(Y.shape[0])
    lintpre = [np.sum(np.abs(X[i])) for i in range(X.shape[0])]
    lintpost = [np.sum(np.abs(Y[i, 0:postmid])) for i in range(Y.shape[0])]

    lpreint = []
    lprepos = []
    lpremxstd = []
    if method == 'integral':
        for i in range(X.shape[0]):
            smax, pos, lint = max_integral(X[i], wlenpre)
            lpreint.append(smax)
            lprepos.append(pos + (wlenpre / 2))
            lint = np.array(lint)
            smax = reduceseg(argrelextrema(lint, np.greater_equal, order=3)[0], 1)
            mxvals = lint[smax]
            lpremxstd.append(np.std(mxvals))
    elif method == 'max':
        for i in range(X.shape[0]):
            smax = np.max(X[i, :])
            pos = np.argmax(X[i, :])
            lpreint.append(smax)
            lprepos.append(pos)
            lpremxstd.append(0)

    lpreint = np.array(lpreint)

    sel = lpreint > 30

    lposint = []
    lpospos = []
    if method == 'integral':
        for i in range(Y.shape[0]):
            smax, pos, _ = max_integral(np.abs(Y[i]), wlenpre)
            lposint.append(smax)
            lpospos.append(pos + (wlenpos / 2))
    elif method == 'max':
        for i in range(Y.shape[0]):
            smax = np.max(Y[i, :])
            if testing:
                if smax > 2:
                    print id[i], i
            pos = np.argmax(Y[i, :])
            lposint.append(smax)
            lpospos.append(pos)

    wtpre = wlenpre * (1000 / freq)
    wtpos = wlenpos * (1000 / freq)
    lprepos = (np.array(lprepos) * (1000 / freq)) - (1500 - (wtpre / 2))
    lpospos = np.array(lpospos) * (1000 / freq) + (off * 1000)

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(15, 15))
    plt.suptitle(title)

    # maxrange = np.max([np.max(lposint), np.max(lpreint)])
    # maxrangeX = np.max(lpreint)
    # maxrangeY = np.max(lposint)

    maxrange = 15
    maxrangeX = 15
    maxrangeY = 15

    for p, lim in enumerate([-1500, -1000, -500]):
        ax1 = fig.add_subplot(3, 4, (p * 4) + 1)
        ax1.axis([-1500, 500, 0, maxrange])
        ax2 = fig.add_subplot(3, 4, (p * 4) + 2)
        ax2.axis([-1500, 500, 0, maxrange])
        ax3 = fig.add_subplot(3, 4, (p * 4) + 3)
        ax3.axis([0, maxrangeX, 0, maxrangeY])
        ax4 = fig.add_subplot(3, 4, (p * 4) + 4)
        ax4.axis([0, maxrangeX, 0, maxrangeY])

        plot_positions(id, lim, lim + 500, [ax1, ax2, ax3, ax4], data=(lprepos, lpreint, lpospos, lposint),
                       mxstd=lpremxstd, tol=tol, eclass=eclass, new=new)

    plt.savefig(data_path + '/prepostdist' + title + ' ' + method + '.pdf', format='pdf')
    plt.show()


def study3(X, Y, ids, title, wlenpre, wlenpos, off=0, freq=0, eclass=True, tol=4, method='max'):
    """
    Study spikes in post events depending of the difference between post and pre event response

    :param X:
    :param Y:
    :param id:
    :param title:
    :param wlenpre:
    :param wlenpos:
    :param off:
    :param freq:
    :param eclass:
    :param tol:
    :param method:
    :return:
    """

    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes

    lpreint = []
    lprepos = []
    lpremxstd = []
    if method == 'integral':
        for i in range(X.shape[0]):
            smax, pos, lint = max_integral(X[i], wlenpre)
            lpreint.append(smax)
            lprepos.append(pos + (wlenpre / 2))
            lint = np.array(lint)
            smax = reduceseg(argrelextrema(lint, np.greater_equal, order=3)[0], 1)
            mxvals = lint[smax]
            lpremxstd.append(np.std(mxvals))
    elif method == 'max':
        for i in range(X.shape[0]):
            smax = np.max(X[i, :])
            pos = np.argmax(X[i, :])
            lpreint.append(smax)
            lprepos.append(pos)
            lpremxstd.append(0)

    lpreint = np.array(lpreint)

    lposint = []
    lpospos = []
    if method == 'integral':
        for i in range(Y.shape[0]):
            smax, pos, _ = max_integral(np.abs(Y[i]), wlenpre)
            lposint.append(smax)
            lpospos.append(pos + (wlenpos / 2))
    elif method == 'max':
        for i in range(Y.shape[0]):
            smax = np.max(Y[i, :])
            pos = np.argmax(Y[i, :])
            lposint.append(smax)
            lpospos.append(pos)

    wtpre = wlenpre * (1000 / freq)

    nsp_plus = []
    nsp_minus = []
    for id, prei, posti in zip(ids, lpreint, lposint):
        vals = col.find_one({'code': id}, {'stmspikes': 1})
        stmspikes = cPickle.loads(vals['stmspikes'])
        vspikes = np.array(stmspikes)

        if stmspikes.shape[0] > 2:
            ldist = []
            for i in range(1, stmspikes.shape[0]):
                ldist.append(stmspikes[i] - stmspikes[i - 1])
            mdist = np.max(np.array(ldist))

            spkcnt = np.zeros(500)
            for i in range(500):
                spkcnt[i] = np.sum(np.logical_and(vspikes >= i, vspikes < (i + 50)))
            spksep = accum_sep(spkcnt)

            if prei > posti:
                nsp_plus.append([mdist, stmspikes.shape[0], stmspikes[-1] - stmspikes[0], spksep])
            else:
                nsp_minus.append([mdist, stmspikes.shape[0], stmspikes[-1] - stmspikes[0], spksep])

    nsp_minus = np.array(nsp_minus)
    nsp_plus = np.array(nsp_plus)

    print nsp_plus.shape
    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(15, 15))
    plt.suptitle(title)
    if nsp_plus.shape[0] > 0:
        ax = fig.add_subplot(422)
        ax.axis([0, 0.5, 0, 10])
        sn.distplot(nsp_plus[:, 0], rug=True, hist=False)
    if nsp_minus.shape[0] > 0:
        ax = fig.add_subplot(421)
        ax.axis([0, 0.5, 0, 10])
        sn.distplot(nsp_minus[:, 0], rug=True, hist=False)

    if nsp_plus.shape[0] > 0:
        ax = fig.add_subplot(424)
        ax.axis([0, 60, 0, 0.05])
        sn.distplot(nsp_plus[:, 1], rug=True, hist=False)
    if nsp_minus.shape[0] > 0:
        ax = fig.add_subplot(423)
        ax.axis([0, 60, 0, 0.05])
        sn.distplot(nsp_minus[:, 1], rug=True, hist=False)

    if nsp_plus.shape[0] > 0:
        ax = fig.add_subplot(426)
        ax.axis([0, 0.5, 0, 10])
        sn.distplot(nsp_plus[:, 2], rug=True, hist=False)
    if nsp_minus.shape[0] > 0:
        ax = fig.add_subplot(425)
        ax.axis([0, 0.5, 0, 10])
        sn.distplot(nsp_minus[:, 2], rug=True, hist=False)

    if nsp_plus.shape[0] > 0:
        ax = fig.add_subplot(428)
        ax.axis([0, 500, 0, 0.1])
        sn.distplot(nsp_plus[:, 3], rug=True, hist=False)
    if nsp_minus.shape[0] > 0:
        ax = fig.add_subplot(427)
        ax.axis([0, 500, 0, 0.1])
        sn.distplot(nsp_minus[:, 3], rug=True, hist=False)

    plt.savefig(data_path + '/spikesdist' + title + ' ' + method + '.pdf', format='pdf')
    plt.show()


def study(X, Y, ids, title, wlenpre, wlenpos):
    """
    Study of mouse events
    
    :param X: 
    :param Y: 
    :return: 
    """
    postmid = int(Y.shape[0])
    lintpre = [np.sum(np.abs(X[i])) for i in range(X.shape[0])]
    lintpost = [np.sum(np.abs(Y[i, 0:postmid])) for i in range(Y.shape[0])]

    lpreint = []
    lprepos = []
    for i in range(X.shape[0]):
        smax = 0
        pos = 0
        for j in range(X.shape[1] - wlenpre):
            sact = np.sum(np.abs(X[i, j:j + wlenpre]))
            if sact > smax:
                smax = sact
                pos = j + (wlenpre / 2)
        lpreint.append(smax)
        lprepos.append(pos)

    lpreint = np.array(lpreint)

    sel = lpreint > 30

    lposint = []
    lpospos = []
    for i in range(Y.shape[0]):
        smax = 0
        pos = 0
        for j in range(X.shape[1] - wlenpos):
            sact = np.sum(np.abs(Y[i, j:j + wlenpos]))
            if sact > smax:
                smax = sact
                pos = j + (wlenpos / 2)
        lposint.append(smax)
        lpospos.append(pos)

    matplotlib.rcParams.update({'font.size': 25})
    fig = plt.figure(figsize=(30, 10))
    plt.suptitle(title)
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
    fig.suptitle(title + ' Wpr= ' + str(wlenpre))
    plt.show()

    fig = plt.figure(figsize=(30, 10))
    fig.suptitle(title + ' Wpr= ' + str(wlenpre) + ' Wpo= ' + str(wlenpos))
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


def make_study2(sttl):
    """
    Analyzes the height of pre and post responses according to the classes of events

    :return:
    """
    method = 'max'  # max integral
    winlen = 10
    X = np.load(data_path + 'mousepre1.npy')
    Y = np.load(data_path + 'mousepost1.npy')
    id = np.load(data_path + 'mouseids1.npy')
    study2(X, Y, id, 'Evento Positivo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=True, tol=4,
           method=method)

    X = np.load(data_path + 'mousepre0.npy')
    Y = np.load(data_path + 'mousepost0.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids0.npy')
    study2(X, Y, id, 'Evento Negativo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False, tol=4,
           method=method, testing=True)

    X = np.load(data_path + 'mousepre2.npy')
    Y = np.load(data_path + 'mousepost2.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids2.npy')
    study2(X, Y, id, 'Evento Intermedio ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False,
           tol=4,
           method=method)


def make_study3(sttl):
    """
    Analizes the statistics of event site spikes depending on the characteristics of the
    pre and post responses

    :return:
    """
    method = 'max'  # max integral
    winlen = 10
    X = np.load(data_path + 'mousepre2.npy')
    Y = np.load(data_path + 'mousepost2.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids2.npy')
    study3(X, Y, id, 'Evento Intermedio ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False,
           tol=4,
           method=method)

    X = np.load(data_path + 'mousepre0.npy')
    Y = np.load(data_path + 'mousepost0.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids0.npy')
    study3(X, Y, id, 'Evento Negativo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False, tol=4,
           method=method)

    X = np.load(data_path + 'mousepre1.npy')
    Y = np.load(data_path + 'mousepost1.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids1.npy')
    study3(X, Y, id, 'Evento Positivo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False, tol=4,
           method=method)


def make_study4(sttl):
    """
    Analizes the statistics of event site spikes depending on the characteristics of the
    pre and post responses

    Study 2 with the new data

    :return:
    """
    method = 'max'  # max integral
    winlen = 10
    X = np.load(data_path + 'mouseprenew2.npy')
    Y = np.load(data_path + 'mousepostnew2.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseidsnew2.npy')
    study2(X, Y, id, 'Evento Intermedio ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False,
           tol=4,
           method=method, new=True)

    X = np.load(data_path + 'mouseprenew0.npy')
    Y = np.load(data_path + 'mousepostnew0.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseidsnew0.npy')
    study2(X, Y, id, 'Evento Negativo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False, tol=4,
           method=method, new=True)

    X = np.load(data_path + 'mouseprenew1.npy')
    Y = np.load(data_path + 'mousepostnew1.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseidsnew1.npy')
    study2(X, Y, id, 'Evento Positivo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False, tol=4,
           method=method, new=True)


def distribution_study(X, Y, wlenpre, method='integral'):
    """
    Extract the maximum value events for distribution comparison
    :return:
    """
    lpreint = []
    if method == 'integral':
        for i in range(X.shape[0]):
            smax, pos, lint = max_integral(X[i], wlenpre)
            lpreint.append(smax)
            lint = np.array(lint)
            smax = reduceseg(argrelextrema(lint, np.greater_equal, order=3)[0], 1)
            mxvals = lint[smax]
    elif method == 'max':
        for i in range(X.shape[0]):
            smax = np.max(X[i, :])
            pos = np.argmax(X[i, :])
            lpreint.append(smax)

    lpreint = np.array(lpreint)

    lposint = []
    if method == 'integral':
        for i in range(Y.shape[0]):
            smax, pos, _ = max_integral(np.abs(Y[i]), wlenpre)
            lposint.append(smax)
    elif method == 'max':
        for i in range(Y.shape[0]):
            smax = np.max(Y[i, :])
            lposint.append(smax)

    lposint = np.array(lposint)

    return lpreint, lposint


def make_study5(sttl):
    """
    Analyzes the distribution of post events

    :param sttl:
    :return:
    """
    method = 'max'  # max integral
    winlen = 10

    X = np.load(data_path + 'mousepre1.npy')
    Y = np.load(data_path + 'mousepost1.npy')
    pre1, pos1 = distribution_study(X, Y, winlen, method=method)

    X = np.load(data_path + 'mousepre2.npy')
    Y = np.load(data_path + 'mousepost2.npy')
    pre2, pos2 = distribution_study(X, Y, winlen, method=method)

    X = np.load(data_path + 'mousepre0.npy')
    Y = np.load(data_path + 'mousepost0.npy')
    pre0, pos0 = distribution_study(X, Y, winlen, method=method)

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(sttl)
    for i, (datapre, datapos, label) in enumerate(zip([pre1, pre0, pre2],
                                                      [pos1, pos0, pos2],
                                                      ['positivo', 'negativo', 'intermedio'])):
        ax1 = fig.add_subplot(3, 3, 3 * i + 1)
        ax1.axis([0, 15, 0, 1])
        ax1.set_xlabel(label)
        sn.distplot(datapos, kde=True, hist=False, rug=True)
        ax2 = fig.add_subplot(3, 3, 3 * i + 3)
        ax2.axis([0, 15, 0, 1])
        ax2.set_xlabel(label + ' POST < PRE')
        datasel = datapos[datapos <= datapre]
        if len(datasel) > 0:
            sn.distplot(datasel, kde=True, hist=False, rug=True)
        ax3 = fig.add_subplot(3, 3, 3 * i + 2)
        ax3.axis([0, 15, 0, 1])
        ax3.set_xlabel(label + ' POST > PRE')
        datasel = datapos[datapos > datapre]
        if len(datasel) > 0:
            sn.distplot(datasel, kde=True, hist=False, rug=True)
        if label in ['negativo', 'intermedio']:
            ax1.text(4, 0.8, 'KS pv = ' + str(ks_2samp(pos1, datapos).pvalue))
            possel = datapos[datapos > datapre]
            ax2.text(4, 0.8, 'KS pv = ' + str(ks_2samp(pos1, possel).pvalue))
            possel = datapos[datapos <= datapre]
            ax3.text(4, 0.8, 'KS pv = ' + str(ks_2samp(pos1, possel).pvalue))

    plt.savefig(data_path + '/posttest' + sttl + ' ' + method + '.pdf', format='pdf')
    plt.show()


def spikes_frequency_graphs(X, Y, ids, title, wlenpre, wlenpos, off=0, freq=0, eclass=True, tol=4, method='max', graph=False):
    """
    Graphs of the spikes frequency (0:200)(200:500)

    :return:
    """
    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes

    lpreint = []
    lprepos = []
    lpremxstd = []
    if method == 'integral':
        for i in range(X.shape[0]):
            smax, pos, lint = max_integral(X[i], wlenpre)
            lpreint.append(smax)
            lprepos.append(pos + (wlenpre / 2))
            lint = np.array(lint)
            smax = reduceseg(argrelextrema(lint, np.greater_equal, order=3)[0], 1)
            mxvals = lint[smax]
            lpremxstd.append(np.std(mxvals))
    elif method == 'max':
        for i in range(X.shape[0]):
            smax = np.max(X[i, :])
            pos = np.argmax(X[i, :])
            lpreint.append(smax)
            lprepos.append(pos)
            lpremxstd.append(0)

    lpreint = np.array(lpreint)

    lposint = []
    lpospos = []
    if method == 'integral':
        for i in range(Y.shape[0]):
            smax, pos, _ = max_integral(np.abs(Y[i]), wlenpre)
            lposint.append(smax)
            lpospos.append(pos + (wlenpos / 2))
    elif method == 'max':
        for i in range(Y.shape[0]):
            smax = np.max(Y[i, :])
            pos = np.argmax(Y[i, :])
            lposint.append(smax)
            lpospos.append(pos)

    wtpre = wlenpre * (1000 / freq)

    nsp_plus = []
    nsp_minus = []
    for id, prei, posti in zip(ids, lpreint, lposint):
        vals = col.find_one({'code': id}, {'stmtime': 1, 'stmspikes': 1})
        stmspikes = cPickle.loads(vals['stmspikes'])
        stmtime = vals['stmtime']
        vspikes = (np.array(stmspikes) - stmtime) * 1000
        if vspikes.shape[0] > 2:
            sp200 = np.sum(vspikes <= 200) * 10.0
            sp500 = np.sum(vspikes > 200) * (10.0 / 3.0)
            sptot = vspikes.shape[0] * 2.0

            if posti > prei:
                nsp_plus.append([sp200, sp500, sptot])
            else:
                nsp_minus.append([sp200, sp500, sptot])

    nsp_minus = np.array(nsp_minus)
    nsp_plus = np.array(nsp_plus)

    # print nsp_plus.shape
    if graph:
        matplotlib.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(15, 15))
        plt.suptitle(title)
        if nsp_plus.shape[0] > 0:
            ax = fig.add_subplot(421)
            ax.axis([0, 250, 0, 0.025])
            ax.set_xlabel('POST > PRE')
            sn.distplot(nsp_plus[:, 0], rug=True, hist=False)
        if nsp_minus.shape[0] > 0:
            ax = fig.add_subplot(422)
            ax.axis([0, 250, 0, 0.025])
            ax.set_xlabel('POST < PRE')
            sn.distplot(nsp_minus[:, 0], rug=True, hist=False)

        if nsp_plus.shape[0] > 0:
            ax = fig.add_subplot(423)
            ax.axis([0, 250, 0, 0.025])
            ax.set_xlabel('POST > PRE')
            sn.distplot(nsp_plus[:, 1], rug=True, hist=False)
        if nsp_minus.shape[0] > 0:
            ax = fig.add_subplot(424)
            ax.axis([0, 250, 0, 0.025])
            ax.set_xlabel('POST < PRE')
            sn.distplot(nsp_minus[:, 1], rug=True, hist=False)

        if nsp_plus.shape[0] > 0:
            ax = fig.add_subplot(425)
            ax.axis([0, 250, 0, 0.025])
            ax.set_xlabel('POST > PRE')
            sn.distplot(nsp_plus[:, 2], rug=True, hist=False)
        if nsp_minus.shape[0] > 0:
            ax = fig.add_subplot(426)
            ax.axis([0, 250, 0, 0.025])
            ax.set_xlabel('POST < PRE')
            sn.distplot(nsp_minus[:, 2], rug=True, hist=False)

        if nsp_plus.shape[0] > 0:
            ax = fig.add_subplot(427)
            ax.axis([0, 250, 0, 250])
            ax.set_xlabel('POST > PRE')
            plt.scatter(nsp_plus[:, 0], nsp_plus[:, 1])
        if nsp_minus.shape[0] > 0:
            ax = fig.add_subplot(428)
            ax.axis([0, 250, 0, 250])
            plt.scatter(nsp_minus[:, 0], nsp_minus[:, 1])
            ax.set_xlabel('POST < PRE')

        plt.savefig(data_path + '/spikesfreq' + title + ' ' + method + '.pdf', format='pdf')
        plt.show()
    return nsp_minus, nsp_plus

def make_study6(sttl):
    """
    Study of spikes frequency

    :return:
    """
    method = 'max'  # max integral
    winlen = 10
    X = np.load(data_path + 'mousepre2.npy')
    Y = np.load(data_path + 'mousepost2.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids2.npy')
    minus2, plus2 = spikes_frequency_graphs(X, Y, id, 'Evento Intermedio ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564,
                            eclass=False, tol=4,
                            method=method)

    X = np.load(data_path + 'mousepre0.npy')
    Y = np.load(data_path + 'mousepost0.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids0.npy')
    minus0, plus0 = spikes_frequency_graphs(X, Y, id, 'Evento Negativo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564,
                            eclass=False, tol=4,
                            method=method)

    X = np.load(data_path + 'mousepre1.npy')
    Y = np.load(data_path + 'mousepost1.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids1.npy')
    minus1, plus1 = spikes_frequency_graphs(X, Y, id, 'Evento Positivo ' + sttl, winlen, winlen, off=0.035, freq=256.4102564102564,
                            eclass=False, tol=4,
                            method=method)

    fig = plt.figure(figsize=(15, 15))
    plt.suptitle('POS vs INT (POS>PRE)')
    ax = fig.add_subplot(321)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200ms')
    sn.distplot(plus1[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(322)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Inter POST > PRE 200ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,0],plus2[:,0]).pvalue))
    sn.distplot(plus2[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(323)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200-500ms')
    sn.distplot(plus1[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(324)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Inter POST > PRE 200-500ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,1],plus2[:,1]).pvalue))
    sn.distplot(plus2[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(325)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 500ms')
    sn.distplot(plus1[:, 2], rug=True, hist=False)
    ax = fig.add_subplot(326)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Inter POST > PRE 200ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,2],plus2[:,2]).pvalue))
    sn.distplot(plus2[:, 2], rug=True, hist=False)
    plt.savefig(data_path + '/spkfreqtestPOSvsINTPOSmxPRE.pdf', format='pdf')
    plt.show()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(321)
    plt.suptitle('POS vs INT (POS<PRE)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200ms')
    sn.distplot(plus1[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(322)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Inter POST < PRE 200ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,0],minus2[:,0]).pvalue))
    sn.distplot(minus2[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(323)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200-500ms')
    sn.distplot(plus1[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(324)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Inter POST < PRE 200-500ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,1],minus2[:,1]).pvalue))
    sn.distplot(minus2[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(325)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 500ms')
    sn.distplot(plus1[:, 2], rug=True, hist=False)
    ax = fig.add_subplot(326)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Inter POST < PRE 500ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,2],minus2[:,2]).pvalue))
    sn.distplot(minus2[:, 2], rug=True, hist=False)
    plt.savefig(data_path + '/spkfreqtestPOSvsINTPOSmnPRE.pdf', format='pdf')
    plt.show()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(321)
    plt.suptitle('POS vs NEG (POS>PRE)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200ms')
    sn.distplot(plus1[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(322)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Neg POST > PRE 200ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,0],plus0[:,0]).pvalue))
    sn.distplot(plus0[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(323)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200-500ms')
    sn.distplot(plus1[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(324)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Neg POST > PRE 300-500ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,1],plus0[:,1]).pvalue))
    sn.distplot(plus0[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(325)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 500ms')
    sn.distplot(plus1[:, 2], rug=True, hist=False)
    ax = fig.add_subplot(326)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Neg POST > PRE 500ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,2],plus0[:,2]).pvalue))
    sn.distplot(plus0[:, 2], rug=True, hist=False)
    plt.savefig(data_path + '/spkfreqtestPOSvsNEGPOSmxPRE.pdf', format='pdf')

    plt.show()

    fig = plt.figure(figsize=(15, 15))
    plt.suptitle('POS vs NEG (POS<PRE)')
    ax = fig.add_subplot(321)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200ms')
    sn.distplot(plus1[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(322)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Neg POST < PRE 200ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,0],minus0[:,0]).pvalue))
    sn.distplot(minus0[:, 0], rug=True, hist=False)
    ax = fig.add_subplot(323)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 200-500ms')
    sn.distplot(plus1[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(324)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Neg POST < PRE 300-500ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,1],minus0[:,1]).pvalue))
    sn.distplot(minus0[:, 1], rug=True, hist=False)
    ax = fig.add_subplot(325)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Normal 500ms')
    sn.distplot(plus1[:, 2], rug=True, hist=False)
    ax = fig.add_subplot(326)
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Neg POST < PRE 500ms')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,2],minus0[:,2]).pvalue))
    sn.distplot(minus0[:, 2], rug=True, hist=False)

    plt.savefig(data_path + '/spkfreqtestPOSvsNEGPOSmnPRE.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    # X = np.load(data_path + 'mousepre2.npy')
    # Y = np.load(data_path + 'mousepost2.npy')
    # id = np.load(data_path + 'mouseids2.npy')
    # make_study2('Orig')
    make_study4('TPS')

    # make_study5('Orig')

    # make_study6('Orig')
