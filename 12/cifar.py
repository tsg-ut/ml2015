#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def label_name(label):
    import cPickle
    fo = open("cifar-10-batches-py/batches.meta", 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict['label_names'][label]

cifar = unpickle(sys.argv[1])
data = cifar['data']

print label_name(cifar['labels'][int(sys.argv[2])])

pic = data[int(sys.argv[2])].reshape(3, 32**2).T.reshape(32, 32, 3)

plt.imshow(pic)
plt.show()
