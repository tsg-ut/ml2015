#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 定義域は(-3, 3)
    xs = np.arange(-3, 3, 0.1)
    # sinカーブに正規分布ノイズをのせる
    ys = np.array([np.sin(x) + np.random.normal(0,0.1) for x in xs])
    plt.plot(xs, ys, 'o')
    # 3次関数フィッティングのパラメータを求める
    param = scipy.polyfit(xs, ys, 3, full=True)[0]
    f = scipy.poly1d(param)
    plt.plot(xs, f(xs))
    plt.show()
