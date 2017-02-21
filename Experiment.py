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

for file in files:
    if file not in ['Exp006']:
        data = Dataset(file)
        data.read()
        data.describe()
