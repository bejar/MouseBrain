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

from flask import Flask, render_template, request
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
from Traffic.Private.DBConfig import mongoconnection
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
    Status de las ciudades
    """
    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes
    vals = col.find({},
                    {'_id': 1, 'exp': 1, 'event': 1, 'label': 1})

    res = {}
    for v in vals:
        if v['exp'] not in res:
            res[v['exp']] = {'ev_cnt': 1, 'labels': {0:0, 1:0}}
            res[v['exp']]['labels'][v['label']] = 1
        else:
            res[v['exp']]['ev_cnt'] += 1
            res[v['exp']]['labels'][v['label']] += 1

    return render_template('ExperimentsList.html', data=res)


@app.route('/Experiment', methods=['GET', 'POST'])
def experiment():
    """
    Status de las ciudades
    """

    payload = request.form['experiment']
    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes
    vals = col.find({'exp': payload},
                    {'_id': 1, 'exp': 1, 'event': 1, 'label': 1})

    res = {}
    for v in vals:
        res[payload + '/' + '%03d' % v['event']] = {'event': v['event'], 'label': v['label']}

    return render_template('EventsList.html', data=res, exp=payload)


@app.route('/View', methods=['GET', 'POST'])
def graphic():
    """
    Generates a page with the training trace

    :return:
    """

    lstyles = ['-', '-', '-', '-'] * 3
    lcolors = ['r', 'g', 'b', 'y'] * 3
    payload = request.form['view']
    exp, event = payload.split('/')
    event = int(event)

    payload = request.form['view']
    client = MongoClient('mongodb://localhost:27017/')
    col = client.MouseBrain.Spikes

    vals = col.find_one({'exp': exp, 'event': event}, {'spike': 1, 'mark': 1, 'event_time': 1, 'pre':1, 'post': 1,
                                                       'vmax':1, 'vmin':1})

    data = cPickle.loads(vals['spike'])
    pre = cPickle.loads(vals['pre'])
    post = cPickle.loads(vals['post'])
    mark = cPickle.loads(vals['mark'])

    disc1 = pre.shape[0]
    disc2 = data.shape[0]-post.shape[0]

    img = StringIO.StringIO()
    fig = plt.figure(figsize=(10, 4), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    axes.axis([0, data.shape[0], vals['vmin'], vals['vmax']])
    axes.set_xlabel('time')
    axes.set_ylabel('num stdv')
    axes.set_title("%s - Event %03d - T=%f" % (exp, event, vals['event_time']))
    maxv = np.max(data)
    minv = np.min(data)

    axes.plot(range(data.shape[0]), data, 'r')
    axes.plot([disc1,disc1], [minv,maxv], 'b')
    axes.plot([disc2,disc2], [minv,maxv], 'b')
    axes.plot([disc1,disc2], [maxv,maxv], 'b')
    axes.plot([disc1,disc2], [minv,minv], 'b')
    if mark[0] !=0:
        axes.plot([mark[0]-5,mark[1]+5], [maxv, maxv], 'g')
        axes.plot([mark[0]-5,mark[1]+5], [minv, minv], 'g')
        axes.plot([mark[0]-5,mark[0]-5], [maxv, minv], 'g')
        axes.plot([mark[1]+5,mark[1]+5], [maxv, minv], 'g')
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue())
    plt.close()

    # if vals is not None:
    #     del vals['_id']
    #
    #     img = StringIO.StringIO()
    #
    #     fig = plt.figure(figsize=(10, 8), dpi=200)
    #     axes = fig.add_subplot(1, 1, 1)
    #
    #     for v, color, style in zip(sorted(vals), lcolors, lstyles):
    #         axes.plot(range(len(vals[v])), vals[v], color + style, label=v)
    #
    #     axes.set_xlabel('epoch')
    #     axes.set_ylabel('acc/loss')
    #     axes.set_title("Training/Test")
    #     axes.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    #     axes.xaxis.set_major_locator(ticker.MultipleLocator(25))
    #
    #     plt.legend()
    #     plt.savefig(img, format='png')
    #     img.seek(0)
    #
    #     plot_url = base64.b64encode(img.getvalue())
    #     plt.close()

    return render_template('SpikeView.html', plot_url=plot_url, exp=exp, event=event)
    # else:
    #     return ""


# @app.route('/Logs')
# def logs():
#     """
#     Returns the logs in the DB
#     """
#     client = MongoClient(mongoconnection.server)
#     db = client[mongoconnection.db]
#     db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
#     col = db[mongoconnection.col]
#
#     vals = col.find({}, {'final_acc': 1, 'final_val_acc': 1, 'time_init': 1, 'time_end': 1, 'time_upd': 1, 'acc': 1,
#                          'done': 1, 'mark': 1, 'config': 1})
#     res = {}
#     for v in vals:
#         if 'time_init' in v:
#             res[v['_id']] = {}
#             if 'mark' in v:
#                 res[v['_id']]['mark'] = v['mark']
#             else:
#                 res[v['_id']]['mark'] = False
#             if 'final_acc' in v:
#                 res[v['_id']]['acc'] = v['final_acc']
#             else:
#                 res[v['_id']]['acc'] = 0
#             if 'final_val_acc' in v:
#                 res[v['_id']]['val_acc'] = v['final_val_acc']
#             else:
#                 res[v['_id']]['val_acc'] = 0
#             res[v['_id']]['init'] = v['time_init']
#             if 'time_end' in v:
#                 res[v['_id']]['end'] = v['time_end']
#             else:
#                 res[v['_id']]['end'] = 'pending'
#             res[v['_id']]['zfactor'] = v['config']['zfactor']
#
#     return render_template('ExperimentsList.html', data=res)
#
#
# @app.route('/Mark', methods=['GET', 'POST'])
# def mark():
#     """
#     Marks an experiment
#     :return:
#     """
#     payload = request.form['mark']
#     client = MongoClient(mongoconnection.server)
#     db = client[mongoconnection.db]
#     db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
#     col = db[mongoconnection.col]
#     vals = col.find_one({'_id': int(payload)}, {'mark': 1, 'done': 1})
#
#     text = ' Not Marked'
#     if vals['done']:
#         if not 'mark' in vals:
#             marked = True
#         else:
#             marked = not vals['mark']
#
#         col.update({'_id': vals['_id']}, {'$set': {'mark': marked}})
#         text = ' Marked'
#
#     head = """
#     <!DOCTYPE html>
# <html>
# <head>
#     <title>Keras NN Mark </title>
#    <meta http-equiv="refresh" content="3;http://%s:%d/Logs" />
#   </head>
# <body>
# """ % (hostname, port)
#     end = '</body></html>'
#
#     return head + str(payload) + text + end
#
#
# @app.route('/Delete', methods=['GET', 'POST'])
# def delete():
#     """
#     Deletes a log
#     """
#     payload = request.form['delete']
#     client = MongoClient(mongoconnection.server)
#     db = client[mongoconnection.db]
#     db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
#     col = db[mongoconnection.col]
#
#     col.remove({'_id': int(payload)})
#
#     head = """
#     <!DOCTYPE html>
# <html>
# <head>
#     <title>Keras NN Delete </title>
#    <meta http-equiv="refresh" content="3;http://%s:%d/Logs" />
#   </head>
# <body>
# """ % (hostname, port)
#     end = '</body></html>'
#
#     return head + str(payload) + ' Removed' + end
#
#

#
# @app.route('/Model', methods=['GET', 'POST'])
# def model():
#     """
#     Generates a page with the configuration of the training and the model
#
#     :return:
#     """
#
#     payload = request.form['model']
#
#     client = MongoClient(mongoconnection.server)
#     db = client[mongoconnection.db]
#     db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
#     col = db[mongoconnection.col]
#
#     vals = col.find_one({'_id': int(payload)}, {'model': 1, 'config': 1, 'svgmodel': 1})
#     pp = pprint.PrettyPrinter(indent=4)
#
#     if 'svgmodel' in vals:
#         svgmodel = vals['svgmodel']
#     else:
#         svgmodel = ''
#
#     head = """
#     <!DOCTYPE html>
# <html>
# <head>
#     <title>Keras NN Config </title>
#   </head>
# <body>
# """
#     end = '</body></html>'
#
#     return head + \
#            '<br><h2>Config:</h2><br><br>' + pprint.pformat(vals['config'], indent=4, width=60).replace('\n', '<br>') + \
#            '<br><br><h2>Graph:</h2><br><br>' + svgmodel + '<br><br><h2>Net:</h2><br><br>' + \
#            pprint.pformat(vals['model'], indent=4, width=40).replace('\n', '<br>') + \
#            '<br>' + \
#            end
#
#
# @app.route('/Report', methods=['GET', 'POST'])
# def report():
#     """
#     Returns a web page with the classification report
#
#     :return:
#     """
#     payload = request.form['report']
#
#     client = MongoClient(mongoconnection.server)
#     db = client[mongoconnection.db]
#     db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
#     col = db[mongoconnection.col]
#
#     vals = col.find_one({'_id': int(payload)}, {'report': 1, 'confusion': 1})
#
#     head = """
#     <!DOCTYPE html>
# <html>
# <head>
#     <title>Keras NN Config </title>
#   </head>
# <body>
# """
#     end = '</body></html>'
#
#     if 'report' in vals:
#         return head + \
#                '<br><h2>Report:</h2><pre>' + vals['report'] + \
#                '</pre><br><br><h2>Confusion:</h2><pre>' + vals['confusion'] + '</pre><br><br>' + \
#                end
#
#     else:
#         return 'No report'
#
#
# @app.route('/Stop', methods=['GET', 'POST'])
# def stop():
#     """
#     Writes on the DB configuration of the process that it has to stop the next epoch
#
#     :return:
#     """
#     payload = request.form['stop']
#
#     client = MongoClient(mongoconnection.server)
#     db = client[mongoconnection.db]
#     db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
#     col = db[mongoconnection.col]
#     col.update({'_id': int(payload)}, {'$set': {'stop': True}})
#
#     head = """
#     <!DOCTYPE html>
# <html>
# <head>
#     <title>Keras NN Stop </title>
#    <meta http-equiv="refresh" content="3;http://%s:%d/Monitor" />
#   </head>
# <body>
# """ % (hostname, port)
#     end = '</body></html>'
#
#     return head + str(payload) + ' Stopped' + end


if __name__ == '__main__':
    # The Flask Server is started
    app.run(host='0.0.0.0', port=port, debug=False)
