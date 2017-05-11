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
from scipy.signal import resample, decimate, resample_poly

__author__ = 'bejar'


class Dataset:
    """
    Class for Config Experiments
    """

    def __init__(self, filename):
        """
        Imports file data
        :param filename:
        """

        self.file = filename
        self.post_marks = None
        self.pre_marks = None
        self.signal = None
        self.orig_signal = None
        self.times = None
        self.events = None
        self.sampling = None
        self.eventsarray = None
        self.orig_eventsarray = None
        self.threshold = None
        self.wbefore = None
        self.wafter = None

    def read(self, normalize=True):
        """
        Read the data from the file
        :return:
        """
        r = Spike2IO(filename=data_path + self.file + '.smr')
        seg = r.read_segment(lazy=False, cascade=True, )
        self.signal = seg.analogsignals[1].rescale('mV').magnitude
        self.orig_signal = seg.analogsignals[1].rescale('mV').magnitude
        self.times = seg.analogsignals[1].times.rescale('s').magnitude
        if len(seg.spiketrains) == 1:
            ind = 0
        else:
            ind = 1
        self.events = seg.spiketrains[ind].times.rescale('s').magnitude
        self.sampling = float(seg.analogsignals[1].sampling_rate.rescale('Hz').magnitude)
        if normalize:
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

    def downsample(self, sampling):
        """
        Resamples the events arrays

        :param sampling:
        :return:
        """
        if self.sampling != sampling:
            factor = self.sampling / sampling
            wlen = int(self.signal.shape[0] / factor)

            # self.signal = downsample(self.signal, wlen)
            # self.times = downsample(self.times, wlen)

            self.signal = resample_poly(self.signal, 10, int(factor * 10))
            self.orig_signal = resample_poly(self.orig_signal, 10, int(factor * 10))
            self.times = resample_poly(self.times, 10, int(factor * 10))

            self.sampling = sampling

    def extract_events(self, before, after):
        """
        Extracts windows for the events with (before)seconds before the event and (after)seconds after the event
        :param before:
        :param after:
        :return:
        """

        def lookup(ipos, event):
            while self.times[ipos] < event:
                ipos += 1
            return ipos

        self.wbefore = int(before * self.sampling)
        self.wafter = int(after * self.sampling)
        cursor = 0
        self.eventsarray = np.zeros((len(self.events), self.wbefore + self.wafter))
        self.orig_eventsarray = np.zeros((len(self.events), self.wbefore + self.wafter))
        fail = []
        for i, ev in enumerate(self.events):
            cursor = lookup(cursor, ev)
            if ((cursor + self.wafter) < self.signal.shape[0]) and ((cursor - self.wbefore) > 0):
                self.eventsarray[i] = np.array(self.signal[cursor - self.wbefore:cursor + self.wafter])
                self.orig_eventsarray[i] = np.array(self.orig_signal[cursor - self.wbefore:cursor + self.wafter])
            else:
                fail.append(i)

        if len(fail) > 0:
            sel = range(self.eventsarray.shape[0])
            for f in fail:
                sel.remove(f)
            self.times = self.times[sel]
            self.events = self.events[sel]
            self.eventsarray = self.eventsarray[sel, :]
            self.orig_eventsarray = self.orig_eventsarray[sel, :]

    def mark_spikes(self, threshold, offset):
        """
        stores the positions of the signal after the event that has a value higher than threshold

        :param threshold:
        :param offset:
        :return:
        """

        self.threshold = threshold
        self.post_marks = np.zeros((len(self.events), 2), dtype=int)
        ahead = int(self.sampling * offset)
        for i in range(len(self.events)):
            for j in range(self.wbefore + ahead, self.eventsarray.shape[1]):
                if self.eventsarray[i, j] > threshold and self.post_marks[i, 0] == 0:
                    self.post_marks[i, 0] = j
                if self.eventsarray[i, j] < threshold and self.post_marks[i, 0] != 0 and self.post_marks[i, 1] == 0:
                    self.post_marks[i, 1] = j
            if self.post_marks[i, 1] == 0 and self.post_marks[i, 0] != 0:
                self.post_marks[i, 1] = self.eventsarray.shape[1] - 5

    def mark_pre_events(self):
        """
        Stores the positions of the before the event that has a value higher than threshold
        :param threshold: 
        :return: 
        """
        if self.threshold is None:
            raise Exception('Threshold not set')
        else:
            threshold = self.threshold
            self.pre_marks = np.zeros((len(self.events), 2), dtype=int)

            for i in range(len(self.events)):
                for j in range(self.wbefore):
                    if self.eventsarray[i, j] > threshold and self.pre_marks[i, 0] == 0:
                        self.pre_marks[i, 0] = j
                    if self.eventsarray[i, j] < threshold and self.pre_marks[i, 0] != 0 and self.pre_marks[i, 1] == 0:
                        self.pre_marks[i, 1] = j
                if self.pre_marks[i, 1] == 0 and self.post_marks[i, 0] != 0:
                    self.pre_marks[i, 1] = self.wbefore

    def assign_labels(self):
        """
        Returns the labels for the spikes (0 or 1 whether there is a spike large enough after the event
        :return:
        """
        if self.post_marks is None or self.pre_marks is None:
            raise Exception('Events not marked')
        else:
            labels = []

            for i in range(len(self.events)):
                if self.post_marks[i,0] == 0:
                    labels.append(0)
                elif self.pre_marks[i,0] == 0:
                    labels.append(1)
                else:
                    labels.append(2)
            return labels

    def show_signal(self, begin=None, end=None):
        """
        Graphic showing a segment (begin, end) of the raw signal
        :param begin:
        :param end:
        :return:
        """

        if begin is None:
            begin = 0
        if end is None:
            end = self.signal.shape[0]

        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(40)

        minaxis = np.min(self.signal)
        maxaxis = np.max(self.signal)
        sp1 = fig.add_subplot(111)
        sp1.plot(self.times[begin:end], self.signal[begin:end], 'r')
        for ev in self.events:
            sp1.plot([ev, ev], [minaxis, maxaxis], 'b')
        plt.show()

    def show_events(self):
        """
        Graphic showing all the signal with the events marked

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
        sp1.plot([self.wbefore, self.wbefore], [minaxis, maxaxis], 'r')
        plt.show()

    def show_event(self, ev):
        """
        Graphic showing a single event

        :param ev:
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
        sp1.plot([self.wbefore, self.wbefore], [minaxis, maxaxis], 'r')
        if self.post_marks is not None and self.post_marks[ev, 0] != 0:
            tm = np.arange(self.post_marks[ev, 0] - 5, self.post_marks[ev, 1] + 5, 1)
            val = np.zeros(self.post_marks[ev, 1] - self.post_marks[ev, 0] + 10) + self.threshold
            sp1.plot(tm, val, 'r')
        plt.show()
        plt.close()

    def get_events_data(self, split=False, discard=None, normalize=False):
        """
        Returns the events data as a matrix
        :param split: Divide the event matrix in pre and post events matrix
        :param discard: Discard a number of seconds before and after the event
        :return:
        """
        if discard is not None:
            vdiscard = int(self.sampling * discard)
        else:
            vdiscard = 0

        lid = [int(self.file[-3:]) * 100 + i for i in range(self.eventsarray.shape[0])]
        prem = self.eventsarray[:, :self.wbefore - vdiscard]
        posm = self.eventsarray[:, self.wbefore + vdiscard:]
        join = np.column_stack((prem, posm))

        if normalize:
            exmn = np.mean(join)
            exstd = np.std(join)
            prem = (prem - exmn) / exstd
            posm = (posm - exmn) / exstd

        if split:
            return prem, posm, lid
        else:
            return np.column_stack((prem, posm)), lid


if __name__ == '__main__':
    data = Dataset('Exp064')
    data.read(normalize=True)
    # data.show_signal(0,5000)
    # data.show_events()
    print(data.sampling)
    data.downsample(99.20634920634922)
    # data.show_signal()

    data.extract_events(1, 0.5)

    # mat = data.get_events_data(discard=0.05)
    #
    # print (mat.shape)

    data.mark_spikes(2, 0.05)

    print data.assign_labels()
    # print data.marks

    for i, e in zip(range(len(data.events)), data.assign_labels()):
        if e == 0:
            data.show_event(i)
