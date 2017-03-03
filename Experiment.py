"""
.. module:: Experiment

Experiment
*************

:Description: Experiment

    

:Authors: bejar
    

:Version: 

:Created on: 21/02/2017 11:37 

"""

from MouseBrain.Data import Dataset
from MouseBrain.Config import data_path
import glob
import numpy as np

__author__ = 'bejar'

if __name__ == '__main__':

    files = sorted([name.split('/')[-1].split('.')[0] for name in glob.glob(data_path + '/*.smr')])

    nev = 0
    split = False
    experiments = []
    experiments2 = []
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

            mat = data.get_events_data(discard=0.01, split=split, normalize=True)

            if split:
               experiments.append(mat[0])
               experiments2.append(mat[1])
            else:
               experiments.append(mat)

    print nev
    if split:
        datamat = np.concatenate(experiments)
        np.save(data_path + 'mousepre.npy', datamat)
        datamat = np.concatenate(experiments2)
        np.save(data_path + 'mousepost.npy', datamat)
    else:
        datamat = np.concatenate(experiments)
        np.save(data_path + 'mouseall.npy', datamat)
