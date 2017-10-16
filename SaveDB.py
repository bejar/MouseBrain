"""
.. module:: SaveDB

SaveDB
*************

:Description: SaveDB

    

:Authors: bejar
    

:Version: 

:Created on: 14/03/2017 9:21 

"""

from pymongo import MongoClient
from MouseBrain.Data import Dataset
from MouseBrain.Config import data_path
import glob
import numpy as np
from collections import Counter
import cPickle
from bson.binary import Binary

__author__ = 'bejar'


if __name__ == '__main__':
    # npath = '/home/bejar/storage/Data/Mouse3/03-04-2017/'
    npath = data_path
    files = sorted([name.split('/')[-1].split('.')[0] for name in glob.glob(npath + '/*.smr')])

    # files = sorted([name.split('/')[-1].split('.')[0] for name in glob.glob(data_path + '/*.smr')])
    client = MongoClient('mongodb://localhost:27017/')
    # client.MouseBrain.Spikes.drop()
    col = client.MouseBrain.Spikes
    nev = 0
    split = True
    experiments = []
    experiments2 = []
    labels = []

    sigma = 2
    latencia = 0.025
    discard = 0.01
    for file in files:
        if file not in ['Exp006'] \
                and (file in ['Exp%03d' % i for i in range(26, 105)]\
                or file in ['Exp%03d' % i for i in range(7, 13)]\
                or file in ['Exp%03d' % i for i in range(16, 21)]):
            print(file)
            data = Dataset(file, path=npath, type='one')
            data.read(normalize=False)
            # data.describe()
            if data.ok and data.sampling > 100.0:
                dsamp = data.sampling
                nev += data.events.shape[0]
                if  data.sampling>257:
                    data.downsample(256.4102564102564)
                # data.show_signal()

                data.extract_events(1.5, 0.5)

                # data.mark_spikes(2, 0.05)
                #
                # # for i in range(len(data.events)):
                # for i in range(10):
                #     data.show_event(i)

                mat = data.get_events_data(discard=latencia + discard, split=True, normalize=True)

                predata = mat[0]
                postdata = mat[1]

                data = Dataset(file, path=npath, type='one')
                data.read(normalize=True)
                if  data.sampling>257:
                    data.downsample(256.4102564102564)
                data.extract_events(1.5, 0.5)
                data.mark_spikes(sigma, latencia + discard)
                data.mark_pre_events()
                data.extract_stimuli_spikes()

                spikes = data.events_array
                ospikes = data.orig_eventsarray
                spostmarks = data.post_marks
                spremarks = data.pre_marks
                labels = data.assign_labels(discard=latencia + discard, threshold=sigma)
                vmax = max(np.max(postdata), np.max(predata))
                vmin = min(np.min(postdata), np.min(predata))
                stmspikes = data.events_spikes

                if data.type == 'one':
                    code = int(file[-3:]) * 100
                else:
                    code = int(file[0:2]) * 10000
                for i in range(spikes.shape[0]):
                    # if labels[i] == 0:
                        event = {'exp': file,
                                 'code': code + i,
                                 'event': i,
                                 'pre': Binary(cPickle.dumps(predata[i], protocol=2)),
                                 'post':  Binary(cPickle.dumps(postdata[i], protocol=2)),
                                 'spike':  Binary(cPickle.dumps(spikes[i], protocol=2)),
                                 'ospike':  Binary(cPickle.dumps(ospikes[i], protocol=2)),
                                 'mark':  Binary(cPickle.dumps(spostmarks[i], protocol=2)),
                                 'premark':  Binary(cPickle.dumps(spremarks[i], protocol=2)),
                                 'label': int(labels[i]),
                                 'stmtime': float(stmspikes[i][0]),
                                 'stmspikes': Binary(cPickle.dumps(stmspikes[i][1], protocol=2)),
                                 'sampling': data.sampling,
                                 'osampling': dsamp,
                                 'wbefore': data.wbefore,
                                 'wafter': data.wafter,
                                 'event_time': float(data.events[i]),
                                 'vmax': vmax,
                                 'vmin': vmin,
                                 'sigma': sigma,
                                 'latency': latencia,
                                 'discard': discard}
                        col.insert(event)