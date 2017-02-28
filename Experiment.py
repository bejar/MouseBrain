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

__author__ = 'bejar'


files = sorted([name.split('/')[-1].split('.')[0] for name in glob.glob(data_path + '/*.smr')])

nev = 0
for file in files:
    if file not in ['Exp006']:
        data = Dataset(file)
        data.read()
        data.describe()
        data.downsample(99.20634920634922)
        data.extract_events(1, 0.5)
        nev += data.events.shape[0]
print(nev)
