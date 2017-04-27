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

    split = True
    experiments = []
    experiments2 = []
    labels = []
    ids = []
    for file in files:
        if file not in ['Exp006'] and file in ['Exp%03d' % i for i in range(30, 73)]:
            # if file in ['Exp019', 'Exp020', 'Exp021', 'Exp022', 'Exp013', 'Exp014', 'Exp015']:
            data = Dataset(file)
            data.read(normalize=False)
            if data.sampling > 100.0:
                data.downsample(256.4102564102564)

                data.extract_events(1.5, 0.5)

                mat = data.get_events_data(discard=0.025, split=split, normalize=True)

                if split:
                    experiments.append(mat[0])
                    experiments2.append(mat[1])
                    lid = mat[2]
                else:
                    experiments.append(mat)
                    lid = mat[1]

                data = Dataset(file)
                data.read(normalize=True)
                data.downsample(256.4102564102564)
                data.extract_events(1.5, 0.5)
                data.mark_spikes(1.75, 0.035)
                labels.append(data.assign_labels())
                ids.append(lid)
    nev = 0

    ldatamat = np.concatenate(labels)

    if split:

        nev += ldatamat[ldatamat == 0].shape[0]
        ids = np.concatenate(ids)
        np.save(save_path + 'mouseids0.npy', ids[ldatamat == 0])
        np.save(save_path + 'mouseids1.npy', ids[ldatamat == 1])

        datamat = np.concatenate(experiments)
        np.save(save_path + 'mousepre0.npy', datamat[ldatamat == 0])
        np.savetxt(save_path + 'mousepre0.csv', datamat[ldatamat == 0], delimiter=';')
        np.save(save_path + 'mousepre1.npy', datamat[ldatamat == 1])
        np.savetxt(save_path + 'mousepre1.csv', datamat[ldatamat == 1], delimiter=';')

        datamat = np.concatenate(experiments2)
        np.save(save_path + 'mousepost0.npy', datamat[ldatamat == 0])
        np.savetxt(save_path + 'mousepost0.csv', datamat[ldatamat == 0], delimiter=';')
        np.save(save_path + 'mousepost1.npy', datamat[ldatamat == 1])
        np.savetxt(save_path + 'mousepost1.csv', datamat[ldatamat == 1], delimiter=';')

    else:
        np.save(save_path + 'mouselabels.npy', ldatamat[ldatamat == 0])
        nev += ldatamat[ldatamat == 0].shape[0]
        datamat = np.concatenate(experiments)
        np.save(save_path + 'mouseall.npy', datamat[ldatamat == 0])

    print nev
