#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import matplotlib.pyplot as plt

dims = [1, 3, 5, 10, 20, 50]

xs = np.arange(-3, 3, 0.1)
ys = np.array([np.sin(x) + np.random.normal(0,0.1) for x in xs])

plt.subplot(231)
f = scipy.poly1d(scipy.polyfit(xs, ys, dims[0], full=True)[0])
plt.scatter(xs, ys, s=3)
plt.plot(xs, f(xs), label="1D")
plt.title("1D")

plt.subplot(232)
f = scipy.poly1d(scipy.polyfit(xs, ys, dims[1], full=True)[0])
plt.scatter(xs, ys, s=3)
plt.plot(xs, f(xs), label="3D")
plt.title("3D")

plt.subplot(233)
f = scipy.poly1d(scipy.polyfit(xs, ys, dims[2], full=True)[0])
plt.scatter(xs, ys, s=3)
plt.plot(xs, f(xs), label="5D")
plt.title("5D")

plt.subplot(234)
f = scipy.poly1d(scipy.polyfit(xs, ys, dims[3], full=True)[0])
plt.scatter(xs, ys, s=3)
plt.plot(xs, f(xs), label="10D")
plt.title("10D")

plt.subplot(235)
f = scipy.poly1d(scipy.polyfit(xs, ys, dims[4], full=True)[0])
plt.scatter(xs, ys, s=3)
plt.plot(xs, f(xs), label="20D")
plt.title("20D")

plt.subplot(236)
f = scipy.poly1d(scipy.polyfit(xs, ys, dims[5], full=True)[0])
plt.scatter(xs, ys, s=3)
plt.plot(xs, f(xs), label="50D")
plt.title("50D")

plt.show()
