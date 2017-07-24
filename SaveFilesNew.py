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

    npath = '/home/bejar/storage/Data/Mouse3/03-04-2017/'
    files = sorted([name.split('/')[-1].split('.')[0] for name in glob.glob(npath + '/*.smr')])

    split = True
    experiments = []
    experiments2 = []
    labels = []
    ids = []
    for file in files:
        # if file in ['Exp019', 'Exp020', 'Exp021', 'Exp022', 'Exp013', 'Exp014', 'Exp015']:
        data = Dataset(file, path=npath, type='two')
        data.read(normalize=False)
        if data.ok and data.sampling > 100.0:
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

            data = Dataset(file, path=npath, type='two')
            data.read(normalize=True)
            data.downsample(256.4102564102564)
            data.extract_events(1.5, 0.5)
            data.mark_spikes(1.75, 0.035)
            data.mark_pre_events()
            labels.append(data.assign_labels())
            ids.append(lid)
    nev = 0

    ldatamat = np.concatenate(labels)

    if split:
        ids = np.concatenate(ids)
        datamat = np.concatenate(experiments)
        datamat2 = np.concatenate(experiments2)
        for dl in np.unique(ldatamat):
            nev += ldatamat[ldatamat == dl].shape[0]
            np.save(save_path + 'mouseidsnew'+str(dl)+'.npy', ids[ldatamat == dl])
            print(ids[ldatamat == dl])

            np.save(save_path + 'mouseprenew'+str(dl)+'.npy', datamat[ldatamat == dl])
            np.savetxt(save_path + 'mouseprenew'+str(dl)+'.csv', datamat[ldatamat == dl], delimiter=';')

            np.save(save_path + 'mousepostnew'+str(dl)+'.npy', datamat2[ldatamat == dl])
            np.savetxt(save_path + 'mousepostnew'+str(dl)+'.csv', datamat2[ldatamat == dl], delimiter=';')


    else:
        np.save(save_path + 'mouselabels.npy', ldatamat[ldatamat == 0])
        nev += ldatamat[ldatamat == 0].shape[0]
        datamat = np.concatenate(experiments)
        np.save(save_path + 'mouseall.npy', datamat[ldatamat == 0])

    print nev
