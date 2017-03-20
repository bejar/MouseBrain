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

    img = StringIO.StringIO()
    fig = plt.figure(figsize=(10, 4), dpi=100)
    axes = fig.add_subplot(2, 1, 1)
    sampling = 1000.0 / float(vals['sampling'])

    axes.axis([- (pre.shape[0] * sampling), data.shape[0]*sampling - (pre.shape[0] * sampling), vals['vmin'], vals['vmax']])
    axes.set_xlabel('time')
    axes.set_ylabel('num stdv')
    axes.set_title("%s - Event %03d - T=%f" % (exp, event, vals['event_time']))
    maxv = np.max(data)
    minv = np.min(data)

    t = np.arange(0.0, data.shape[0])*sampling - (pre.shape[0] * sampling)
    axes.xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(2))
    disc2 = int(vals['discard'] * 1000.0)
    axes.plot(t, data, 'r')
    axes.plot([0,0], [minv,maxv], 'b')
    axes.plot([disc2,disc2], [minv,maxv], 'b')
    axes.plot([0,disc2], [maxv,maxv], 'b')
    axes.plot([0,disc2], [minv,minv], 'b')

    ltn = int(vals['latency'] * 1000.0)
    axes.plot([ltn, ltn], [maxv,minv], 'c')

    if mark[0] != 0:
        mark[0] = int(mark[0] * sampling) - (pre.shape[0] * sampling)
        mark[1] = int(mark[1] * sampling) - (pre.shape[0] * sampling)
        axes.plot([mark[0],mark[1]], [maxv, maxv], 'g')
        axes.plot([mark[0],mark[1]], [minv, minv], 'g')
        axes.plot([mark[0],mark[0]], [maxv, minv], 'g')
        axes.plot([mark[1],mark[1]], [maxv, minv], 'g')

    axes.plot([0,data.shape[0]*sampling - (pre.shape[0] * sampling)], [vals['sigma'],vals['sigma'] ], 'y')
    # plt.legend()

    maxv = np.max(odata)
    minv = np.min(odata)
    axes2 = fig.add_subplot(2, 1, 2)
    axes2.axis([- (pre.shape[0] * sampling), data.shape[0]*sampling - (pre.shape[0] * sampling), minv, maxv])
    axes2.set_xlabel('time')
    axes2.set_ylabel('mV')

    axes2.xaxis.set_major_locator(ticker.MultipleLocator(100))

    t = np.arange(0.0, data.shape[0])*sampling - (pre.shape[0] * sampling)
    axes2.plot(t, odata, 'r')
    axes2.plot([0,0], [minv,maxv], 'b')
    axes2.plot([ltn, ltn], [maxv,minv], 'c')

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
