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
from MouseBrain.Config import data_path, save_path
import glob
import numpy as np
from collections import Counter

__author__ = 'bejar'

if __name__ == '__main__':

    files = sorted([name.split('/')[-1].split('.')[0] for name in glob.glob(data_path + '/*.smr')])

    nev = 0
    split = True
    experiments = []
    experiments2 = []
    labels = []
    for file in files:
        if file not in ['Exp006']:
            data = Dataset(file)
            data.read(normalize=False)
            if data.sampling > 100.0:
                nev += data.events.shape[0]
                data.downsample(256.4102564102564)

                data.extract_events(2, 0.25)

                mat = data.get_events_data(discard=0.01, split=split, normalize=True)

                if split:
                   experiments.append(mat[0])
                   experiments2.append(mat[1])
                else:
                   experiments.append(mat)

                data = Dataset(file)
                data.read(normalize=True)
                data.downsample(256.4102564102564)
                data.extract_events(2, 0.25)
                data.mark_spikes(1.5, 0.035)
                labels.append(data.assign_labels())
    print nev
    if split:
        ldatamat = np.concatenate(labels)
        np.save(save_path + 'mouselabels.npy', ldatamat[ldatamat==0])
        datamat = np.concatenate(experiments)
        np.save(save_path + 'mousepre.npy', datamat[ldatamat==0])
        datamat = np.concatenate(experiments2)
        np.save(save_path + 'mousepost.npy', datamat[ldatamat==0])

    else:
        ldatamat = np.concatenate(labels)
        np.save(save_path + 'mouselabels.npy', ldatamat[ldatamat==0])
        datamat = np.concatenate(experiments)
        np.save(save_path + 'mouseall.npy', datamat[ldatamat==0])


