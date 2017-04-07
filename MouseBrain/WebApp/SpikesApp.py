"""
.. module:: WebMonitor

ConvoTest
*************

:Description: WebStatus



:Authors: bejar


:Version:

:Created on: 28/11/2016 11:10

"""

import socket

from flask import Flask, render_template, request, url_for, redirect
from pymongo import MongoClient
import StringIO

# import bokeh.plotting as plt

import matplotlib
import cPickle
from bson.binary import Binary
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import base64
import seaborn as sns
import numpy as np
import pprint
import time

__author__ = 'bejar'

# Configuration stuff
hostname = socket.gethostname()
port = 9000

app = Flask(__name__)


@app.route('/MouseBrain')
def info():
    """
    Lista de experimentos
    """
    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes
    vals = col.find({},
                    {'_id': 1, 'exp': 1, 'event': 1, 'label': 1, 'check': 1, 'annotation':1})
    res = {}
    for v in vals:
        if v['exp'] not in res:
            res[v['exp']] = {'ev_cnt': 1, 'labels': {0:0, 1:0}}
            res[v['exp']]['labels'][v['label']] = 1
        else:
            res[v['exp']]['ev_cnt'] += 1
            res[v['exp']]['labels'][v['label']] += 1
        if 'annotation' in v and v['annotation'] != '':
            res[v['exp']]['check'] = True

    return render_template('ExperimentsList.html', data=res)


@app.route('/Experiment/<exp>', methods=['GET', 'POST'])
def experiment(exp):
    """
    Experimento
    """

    payload = exp #request.form['experiment']
    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes
    vals = col.find({'exp': payload},
                    {'_id': 1, 'exp': 1, 'event': 1, 'label': 1, 'check': 1, 'annotation': 1})

    res = {}
    for v in vals:
        if 'check' in v:
            mark = v['check']
        else:
            mark = False
        annotation = ''
        if 'annotation' in v:
            annotation = v['annotation']
        res['%03d' % v['event']] = {'event': v['event'], 'label': v['label'], 'check': mark, 'annotation': annotation }

    return render_template('EventsList.html', data=res, exp=payload, port=port)


@app.route('/View/<exp>/<event>', methods=['GET', 'POST'])
def graphic(exp, event):
    """
    Generates a page with the training trace

    :return:
    """

    # payload = request.form['view']
    # exp, event = payload.split('/')
    event = int(event)

    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes

    vals = col.find_one({'exp': exp, 'event': event}, {'spike': 1, 'ospike': 1, 'mark': 1, 'event_time': 1, 'pre':1, 'post': 1,
                                                       'vmax':1, 'vmin':1, 'sampling':1, 'sigma':1, 'latency': 1,
                                                       'discard': 1, 'annotation': 1})

    data = cPickle.loads(vals['spike'])
    odata = cPickle.loads(vals['ospike'])
    pre = cPickle.loads(vals['pre'])
    post = cPickle.loads(vals['post'])
    mark = cPickle.loads(vals['mark'])
    nrows = 6

    img = StringIO.StringIO()
    fig = plt.figure(figsize=(10, 16), dpi=100)

    axes = fig.add_subplot(nrows, 1, 1)
    sampling = 1000.0 / float(vals['sampling'])

    axes.axis([- (pre.shape[0] * sampling), data.shape[0]*sampling - (pre.shape[0] * sampling), vals['vmin'], vals['vmax']])
    axes.set_xlabel('time')
    axes.set_ylabel('num stdv')
    axes.set_title("%s - Event %03d - T=%f" % (exp, event, vals['event_time']))
    maxvg = np.max(data)
    minvg = np.min(data)

    t = np.arange(0.0, data.shape[0])*sampling - (pre.shape[0] * sampling)
    axes.xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(2))
    disc2 = int(vals['discard'] * 1000.0)
    axes.plot(t, data, 'r')
    axes.plot([0,0], [minvg,maxvg], 'b')
    axes.plot([disc2,disc2], [minvg,maxvg], 'b')
    axes.plot([0,disc2], [maxvg,maxvg], 'b')
    axes.plot([0,disc2], [minvg,minvg], 'b')

    ltn = int(vals['latency'] * 1000.0)
    axes.plot([ltn, ltn], [maxvg,minvg], 'c')

    if mark[0] != 0:
        mark[0] = int(mark[0] * sampling) - (pre.shape[0] * sampling)
        mark[1] = int(mark[1] * sampling) - (pre.shape[0] * sampling)
        axes.plot([mark[0],mark[1]], [maxvg, maxvg], 'g')
        axes.plot([mark[0],mark[1]], [minvg, minvg], 'g')
        axes.plot([mark[0],mark[0]], [maxvg, minvg], 'g')
        axes.plot([mark[1],mark[1]], [maxvg, minvg], 'g')
    axes.plot([0,data.shape[0]*sampling - (pre.shape[0] * sampling)], [vals['sigma'],vals['sigma'] ], 'y')
    # plt.legend()

    maxv = np.max(odata)
    minv = np.min(odata)
    axes2 = fig.add_subplot(nrows, 1, 2)
    axes2.axis([- (pre.shape[0] * sampling), data.shape[0]*sampling - (pre.shape[0] * sampling), minv, maxv])
    axes2.set_xlabel('time')
    axes2.set_ylabel('mV')
    axes2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    t = np.arange(0.0, data.shape[0])*sampling - (pre.shape[0] * sampling)
    axes2.plot(t, odata, 'r')
    axes2.plot([0,0], [minv,maxv], 'b')
    axes2.plot([ltn, ltn], [maxv,minv], 'c')

    axes3 = fig.add_subplot(nrows, 2, 5)
    axes3.axis([0,  (pre.shape[0] * sampling), minvg, maxvg])
    t = np.arange(0.0, pre.shape[0]) * sampling
    axes3.plot(t, pre)

    # X ms windows integral
    wlenpre = 25
    smax = 0
    pos = 0
    for j in range(pre.shape[0]-wlenpre):
        sact = np.sum(np.abs(pre[j:j+wlenpre]))
        if sact > smax:
            smax = sact
            pos = j

    axes3.plot([pos * sampling, pos * sampling], [maxvg,minvg], 'r')
    axes3.plot([(pos+wlenpre) * sampling, (pos+wlenpre) * sampling], [maxvg,minvg], 'r')

    axes4 = fig.add_subplot(nrows, 2, 6)
    axes4.axis([0,  (post.shape[0] * sampling), minvg, maxvg])

    t = np.arange(0.0, post.shape[0]) * sampling
    axes4.plot(t, post)

    # X ms windows integral
    # wlenpre = 40
    smax = 0
    pos = 6
    for j in range(6, post.shape[0]-wlenpre):
        sact = np.sum(np.abs(post[j:j+wlenpre]))
        if sact > smax:
            smax = sact
            pos = j
    axes4.plot([pos * sampling, pos * sampling], [maxvg,minvg], 'r')
    axes4.plot([(pos + wlenpre) * sampling, (pos + wlenpre) * sampling], [maxvg,minvg], 'r')

    from scipy.signal import hilbert

    axes5 = fig.add_subplot(nrows, 2, 7)
    a_pre = hilbert(pre)
    envpre = np.abs(a_pre)
    maxv = np.max(envpre)
    minv = np.min(envpre)

    axes5.axis([0,  (pre.shape[0] * sampling), minv, maxv])
    t = np.arange(0.0, pre.shape[0]) * sampling
    axes5.plot(t, envpre)

    axes6 = fig.add_subplot(nrows, 2, 8)
    a_post = hilbert(post)
    envpost = np.abs(a_post)
    maxv = np.max(envpost)
    minv = np.min(envpost)

    axes6.axis([0,  (post.shape[0] * sampling), minv, maxv])

    t = np.arange(0.0, post.shape[0]) * sampling
    axes6.plot(t, envpost)

    axes7 = fig.add_subplot(nrows, 2, 9)
    ip_pre = np.unwrap(np.angle(a_pre))
    if_pre = np.diff(ip_pre)/(2*np.pi)*sampling
    maxv = np.max(if_pre)
    minv = np.min(if_pre)
    axes7.axis([0,  (pre.shape[0] * sampling), minv, maxv])
    t = np.arange(0.0, pre.shape[0]) * sampling
    axes7.plot(t[1:], if_pre)

    axes8 = fig.add_subplot(nrows, 2, 10)
    ip_post = np.unwrap(np.angle(a_post))
    if_post = np.diff(ip_post)/(2*np.pi)*sampling
    maxv = np.max(if_post)
    minv = np.min(if_post)
    axes8.axis([0,  (post.shape[0] * sampling), minv, maxv])
    t = np.arange(0.0, post.shape[0]) * sampling
    axes8.plot(t[1:], if_post)

    axes9 = fig.add_subplot(nrows, 2, 11)
    axes9.plot(np.real(a_pre), np.imag(a_pre))

    axes10 = fig.add_subplot(nrows, 2, 12)
    axes10.plot(np.real(a_post), np.imag(a_post))

    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue())
    plt.close()

    annotation = '' if 'annotation' not in vals else vals['annotation']
    return render_template('SpikeView.html', plot_url=plot_url, exp=exp, event=event, ann=annotation, port=port)


@app.route('/Annotate/<exp>/<event>', methods=['GET', 'POST'])
def annotate(exp, event):
    """
    Annotation of events
    :return:
    """
    payload = request.form['annotation']

    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes
    vals = col.find_one({'exp': exp, 'event': int(event)}, {'_id':1, 'check': 1})
    col.update({'_id': vals['_id']}, {'$set': {'annotation': payload}})

    return redirect(url_for('.graphic', exp=exp, event=event))


@app.route('/Mark/<exp>/<event>', methods=['GET', 'POST'])
def mark(exp, event):
    """
    Marks an experiment
    :return:
    """
    # payload = request.form['mark']
    # exp, event = payload.split('/')
    event = int(event)

    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes

    vals = col.find_one({'exp': exp, 'event': event}, {'_id':1, 'check': 1})
    if not 'check' in vals:
        marked = True
    else:
        marked = not vals['check']

    col.update({'_id': vals['_id']}, {'$set': {'check': marked}})

    return redirect(url_for('.experiment', exp=exp))

if __name__ == '__main__':
    # The Flask Server is started
    app.run(host='0.0.0.0', port=port, debug=False)
