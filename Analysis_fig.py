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
    ax1, ax3 = axes
    ax1.plot([0, 0], [0, 180], 'k')
    # ax2.plot([0, 0], [0, 180], 'k')
    lprepos, lpreint, lpospos, lposint = data
    for ip, (prep, prei, posp, posi, idi) in enumerate(zip(lprepos, lpreint, lpospos, lposint, id)):
        mark = 's' if new and ((idi / 1000) < 100) else 'o'
        if eleft < prep < eright:
            if (eclass and prei > posi) or (not eclass and prei < posi):
                pass
                # ax1.plot([prep, posp], [prei, posi], 'g:')
                #
                # clm = 'b' if mark == 'o' else 'c'
                # cm = 'k' if mxstd[ip] > tol else clm
                # ax1.plot([prep], [prei], cm, marker=mark)
                # ax1.plot([posp], [posi], clm, marker=mark)
                # # print(id[ip], prei, posi)
                # ax3.plot(posi, prei, clm, marker=mark)
                # if prei >= posi:
                #     ax3.set_xlabel('POST < PRE')
                # else:
                #     ax3.set_xlabel('POST > PRE')
            else:
                clm = 'y' if mark == 'o' else 'g'
                cm = 'k' if mxstd[ip] > tol else clm
                ax1.plot([prep], [prei], cm, marker=mark)
                ax1.plot([posp], [posi], clm, marker=mark)
                ax1.plot([prep, posp], [prei, posi], 'r:')
                ax3.plot(posi, prei, clm, marker=mark)
                # if prei >= posi:
                #     ax4.set_xlabel('POST < PRE')
                # else:
                #     ax4.set_xlabel('POST > PRE')


def study2(X, Y, id, title, wlenpre, wlenpos, off=0, freq=0, eclass=True, tol=4, method='integral', new=False, nfile=''):
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
            pos = np.argmax(Y[i, :])
            lposint.append(smax)
            lpospos.append(pos)

    wtpre = wlenpre * (1000 / freq)
    wtpos = wlenpos * (1000 / freq)
    lprepos = (np.array(lprepos) * (1000 / freq)) - (1500 - (wtpre / 2))
    lpospos = np.array(lpospos) * (1000 / freq) + (off * 1000)

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(15, 15))
    # plt.suptitle(title)

    # maxrange = np.max([np.max(lposint), np.max(lpreint)])
    # maxrangeX = np.max(lpreint)
    # maxrangeY = np.max(lposint)

    maxrange = 15
    maxrangeX = 15
    maxrangeY = 15
    sn.set(style="whitegrid",  color_codes=True)
    for p, lim in enumerate([-1500, -1000, -500]):
        ax1 = fig.add_subplot(3, 2, (p * 2) + 1)
        ax1.axis([-1500, 500, 0, maxrange])
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Stdev')
        # ax2 = fig.add_subplot(3, 2, (p * 4) + 2)
        # ax2.axis([-1500, 500, 0, maxrange])
        ax3 = fig.add_subplot(3, 2, (p * 2) + 2)
        ax3.axis([0, maxrangeX, 0, maxrangeY])
        ax3.set_xlabel('Stdev')
        ax3.set_ylabel('Stdev')
        # ax4 = fig.add_subplot(3, 2, (p * 4) + 4)
        # ax4.axis([0, maxrangeX, 0, maxrangeY])

        plot_positions(id, lim, lim + 500, [ax1, ax3], data=(lprepos, lpreint, lpospos, lposint),
                       mxstd=lpremxstd, tol=tol, eclass=eclass, new=new)

    plt.savefig(data_path + '/PeakMaxDist' + nfile + '.pdf', format='pdf')
    plt.show()

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
    study2(X, Y, id, 'Positive Events ', winlen, winlen, off=0.035, freq=256.4102564102564, eclass=True, tol=4,
           method=method, nfile='POS')

    X = np.load(data_path + 'mousepre0.npy')
    Y = np.load(data_path + 'mousepost0.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids0.npy')
    study2(X, Y, id, 'Negative Events', winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False, tol=4,
           method=method, nfile='NEG')

    X = np.load(data_path + 'mousepre2.npy')
    Y = np.load(data_path + 'mousepost2.npy')
    print(X.shape, Y.shape)
    id = np.load(data_path + 'mouseids2.npy')
    study2(X, Y, id, 'Intermediate Events ', winlen, winlen, off=0.035, freq=256.4102564102564, eclass=False,
           tol=4, method=method, nfile='INT')


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

    print 'Positive:', plus1.shape[0]
    print 'Negative:', minus0.shape[0]
    print 'Intermediate:', minus2.shape[0]

    sn.set(style="whitegrid",  color_codes=True)
    fig = plt.figure(figsize=(15, 15))

    ax = sn.distplot(plus1[:, 0], rug=True, hist=False)
    plt.suptitle('Positive events (200ms)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Frequency (Hz)')

    plt.savefig(data_path + '/spkfreqtestPOS200.pdf', format='pdf')
    plt.show()


    ax = sn.distplot(minus2[:, 0], rug=True, hist=False)
    plt.suptitle('Intermediate events (200ms)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Frequency (Hz)')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,0],minus2[:,0]).pvalue))

    plt.savefig(data_path + '/spkfreqtestINT200.pdf', format='pdf')
    plt.show()


    ax = sn.distplot(plus1[:, 2], rug=True, hist=False)
    plt.suptitle('Positive events (500ms)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Frequency (Hz)')

    plt.savefig(data_path + '/spkfreqtestPOS500.pdf', format='pdf')
    plt.show()

    ax = sn.distplot(minus2[:, 2], rug=True, hist=False)
    plt.suptitle('Intermediate events (500ms)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Frequency (Hz)')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,2],minus2[:,2]).pvalue))

    plt.savefig(data_path + '/spkfreqtestINT500.pdf', format='pdf')
    plt.show()


    ax = sn.distplot(minus0[:, 0], rug=True, hist=False)
    plt.suptitle('Negative events (200ms)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Frequency (Hz)')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,0],minus0[:,0]).pvalue))

    plt.savefig(data_path + '/spkfreqtestNEG200.pdf', format='pdf')
    plt.show()

    ax = sn.distplot(minus0[:, 2], rug=True, hist=False)
    plt.suptitle('Negative events (500ms)')
    ax.axis([0, 250, 0, 0.025])
    ax.set_xlabel('Frequency (Hz)')
    ax.text(100, 0.02, 'KS pv = ' + str(ks_2samp(plus1[:,2],minus0[:,2]).pvalue))
    plt.savefig(data_path + '/spkfreqtestNEG500.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    make_study2('Orig')

    # make_study6('Orig')
