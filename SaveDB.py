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

    files = sorted([name.split('/')[-1].split('.')[0] for name in glob.glob(data_path + '/*.smr')])
    client = MongoClient('mongodb://localhost:27017/')
    client.MouseBrain.Spikes.drop()
    col = client.MouseBrain.Spikes
    nev = 0
    split = True
    experiments = []
    experiments2 = []
    labels = []
    for file in files:
        if file not in ['Exp006']:
            data = Dataset(file)
            data.read(normalize=False)
            # data.describe()
            nev += data.events.shape[0]
            data.downsample(99.20634920634922)
            # data.show_signal()

            data.extract_events(1, 0.5)

            # data.mark_spikes(2, 0.05)
            #
            # # for i in range(len(data.events)):
            # for i in range(10):
            #     data.show_event(i)

            mat = data.get_events_data(discard=0.05, split=True, normalize=True)

            predata = mat[0]
            postdata = mat[1]

            data = Dataset(file)
            data.read(normalize=True)
            data.downsample(99.20634920634922)
            data.extract_events(1, 0.25)
            data.mark_spikes(2, 0.05)
            spikes = data.eventsarray
            spmarks = data.marks
            labels = data.assign_labels()
            vmax = max(np.max(postdata), np.max(predata))
            vmin = min(np.min(postdata), np.min(predata))

            for i in range(spikes.shape[0]):
                event = {'exp': file,
                         'event': i,
                         'pre': Binary(cPickle.dumps(predata[i], protocol=2)),
                         'post':  Binary(cPickle.dumps(postdata[i], protocol=2)),
                         'spike':  Binary(cPickle.dumps(spikes[i], protocol=2)),
                         'mark':  Binary(cPickle.dumps(spmarks[i], protocol=2)),
                         'label': int(labels[i]),
                         'sampling': data.sampling,
                         'wbefore': data.wbefore,
                         'wafter': data.wafter,
                         'event_time': float(data.events[i]),
                         'vmax': vmax,
                         'vmin': vmin}

                col.insert(event)

