"""
.. module:: Misc

Misc
*************

:Description: Misc

    

:Authors: bejar
    

:Version: 

:Created on: 27/04/2017 8:45 

"""

import numpy as np

__author__ = 'bejar'


def reduceseg(lval, tol):
    """
    Eliminates from the list all the elements that are at a distance less than tol from the next
    :param lval: 
    :param tol: 
    :return: 
    """
    res = [lval[0]]
    curr = lval[0]
    for i in range(1,len(lval)):
        if lval[i] - curr > tol:
            res.append(lval[i])
        curr = lval[i]

    return res


def max_integral(signal, wlen):
    """
    Return the integral and the position of the window with maximal integral
    integral = sum of the absolute value of the signal  
     
    :param wlen: 
    :return: 
    """

    smax = 0
    pos = 0
    lint = []
    for j in range(signal.shape[0] - wlen):
        sact = np.sum(np.abs(signal[j:j + wlen]))
        lint.append(sact)
        if sact > smax:
            smax = sact
            pos = j

    return smax, pos, lint

if __name__ == '__main__':
    l = [1,2,3,4,5,6,7, 9, 11, 12,13, 17, 19, 20]
    print(reduceseg(l,1))