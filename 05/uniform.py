#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import matplotlib.pyplot as plt

N = 1000

for n in np.arange(1, 16, 2):
    n = int(n)
    # リスト内包表記
    xs = [sum([random.choice(np.arange(1, 7)) for _ in range(n)]) for _ in range(N)]
    plt.clf()
    # normed=Trueでヒストグラムが正規化される
    plt.hist(xs, bins=max(xs), normed=True)
    plt.xlim([0, 100])
    plt.ylim([0, 0.3])
    plt.suptitle('n = {}'.format(n))
    plt.savefig('fig{}.png'.format(n))
