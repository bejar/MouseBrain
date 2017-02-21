"""
.. module:: Dataset

Dataset
*************

:Description: Dataset

    

:Authors: bejar
    

:Version: 

:Created on: 20/02/2017 13:45 

"""

from neo.io import Spike2IO
import numpy as np
from MouseBrain.Config.Constants import data_path
import matplotlib.pyplot as plt

__author__ = 'bejar'


class Dataset():
    """
    Class for Config Experiments
    """

    def __init__(self, filename):
        """
        Imports file data
        :param filename:
        """

        self.file = filename
        self.marks = None
        self.signal = None
        self.times = None
        self.events = None
        self.sampling = None
        self.eventsarray = None
        self.threshold = None


    def read(self):
        """
        Read the data from the file
        :return:
        """
        r = Spike2IO(filename=data_path + self.file + '.smr')
        seg = r.read_segment(lazy=False, cascade=True,)
        self.signal = seg.analogsignals[1].rescale('mV').magnitude
        self.times = seg.analogsignals[1].times.rescale('s').magnitude
        if len(seg.spiketrains) == 1:
            ind = 0
        else:
            ind = 1
        self.events = seg.spiketrains[ind].times.rescale('s').magnitude
        self.sampling = seg.analogsignals[1].sampling_rate.rescale('Hz').magnitude
        self.signal -= np.mean(self.signal)
        self.signal /= np.std(self.signal)



    def describe(self):
        """
        Some infor about the dataset
        :return:
        """
        print ('File =', self.file)
        print ('Sampling = ', self.sampling)
        print ('Num Samples = ', self.signal.shape)
        print ('Num Stimuli = ', self.events.shape)
        print ('----------------------------------')

    def extract_events(self, before, after):
        """
        Extracts windows for the events with (before)s before the event and (after)s after the event
        :param before:
        :param after:
        :return:
        """
        def lookup(ipos, ev):
            while self.times[ipos] < ev:
                ipos += 1
            return ipos

        self.wbefore = int(before * self.sampling)
        self.wafter = int(after * self.sampling)
        cursor = 0
        self.eventsarray = np.zeros((len(self.events), self.wbefore+self.wafter))
        for i, ev in enumerate(self.events):
            cursor = lookup(cursor, ev)
            # print cursor
            # print self.eventsarray.shape[1], cursor-self.wbefore, cursor+self.wafter
            # if ((cursor+self.wafter) < self.eventsarray.shape[1]) and ((cursor-self.wbefore) > 0):
            #     print self.eventsarray.shape[1], cursor-self.wbefore, cursor+self.wafter
            self.eventsarray[i] = np.array(self.signal[cursor-self.wbefore:cursor+self.wafter])

    def mark_spikes(self, threshold):
        """
        stores the positions of the signal after the event that has a value higher than threshold

        :return:
        """

        self.threshold = threshold
        self.marks = np.zeros((len(self.events), 2), dtype=int)
        ahead = int(self.sampling*0.1)
        for i in range(len(self.events)):
            for j in range(self.wbefore+ahead, self.eventsarray.shape[1]):
                if self.eventsarray[i,j] > threshold and self.marks[i,0] == 0:
                    self.marks[i,0] = j
                if self.eventsarray[i,j] < threshold and self.marks[i,0] != 0 and self.marks[i,1] == 0:
                    self.marks[i,1] = j
            if self.marks[i,1] == 0:
                self.marks[i,1] = self.eventsarray.shape[1]-5



    def show_signal(self, begin, end):
        """

        :param begin:
        :param end:
        :return:
        """
        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(40)

        minaxis = np.min(self.signal)
        maxaxis = np.max(self.signal)
        num = end - begin
        sp1 = fig.add_subplot(111)
        # sp1.axis([0, num, minaxis, maxaxis])
        sp1.plot(self.times[begin:end], self.signal[begin:end], 'r')
        for ev in self.events:
            sp1.plot([ev,ev], [minaxis, maxaxis], 'b')
        plt.show()

    def show_events(self):
        """

        :return:
        """
        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(40)

        minaxis = np.min(self.eventsarray)
        maxaxis = np.max(self.eventsarray)
        num = self.eventsarray.shape[1]
        sp1 = fig.add_subplot(111)
        sp1.axis([0, num, minaxis, maxaxis])
        t = np.arange(0.0, num, 1)
        for i in range(len(self.events)):
            sp1.plot(t, self.eventsarray[i], 'b')
        sp1.plot([self.wbefore,self.wbefore], [minaxis, maxaxis], 'r')
        plt.show()

    def show_event(self, ev, marks=False):
        """

        :return:
        """
        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(40)

        minaxis = np.min(self.eventsarray)
        maxaxis = np.max(self.eventsarray)
        num = self.eventsarray.shape[1]
        sp1 = fig.add_subplot(111)
        sp1.axis([0, num, minaxis, maxaxis])
        t = np.arange(0.0, num, 1)
        sp1.plot(t, self.eventsarray[ev], 'b')
        sp1.plot([self.wbefore,self.wbefore], [minaxis, maxaxis], 'r')
        if self.marks is not None and self.marks[ev,0] != 0:
            tm = np.arange(self.marks[ev,0]-5, self.marks[ev,1]+5, 1)
            val = np.zeros(self.marks[ev,1] - self.marks[ev,0] + 10) + self.threshold
            sp1.plot(tm, val, 'r')
        plt.show()


if __name__ == '__main__':
    data = Dataset('Exp034')
    data.read()
    # data.show_signal(0,5000)
    # data.show_events()
    data.extract_events(1, 0.5)

    data.mark_spikes(2)

    for i in range(len(data.events)):
        data.show_event(i, marks=True)
