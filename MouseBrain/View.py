"""
.. module:: View

View
*************

:Description: View

    

:Authors: bejar
    

:Version: 

:Created on: 03/03/2017 8:55 

"""

__author__ = 'bejar'


from MouseBrain.Config import data_path
import numpy as np
import matplotlib.pyplot as plt


def show_event(events, ev):
    """
    Graphic showing a single event

    :param ev:
    :return:
    """
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)

    minaxis = np.min(events)
    maxaxis = np.max(events)
    num = events.shape[1]
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    sp1.plot(t, events[ev], 'b')

    plt.show()
    plt.close




if __name__ == '__main__':
    events = np.load(data_path + 'mouseall.npy')
    for i in range(events.shape[0]):
        show_event(events, i)