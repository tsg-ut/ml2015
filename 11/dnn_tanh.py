#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt

def flatten(l):
    from itertools import chain
    return list(chain.from_iterable(l))

class Perceptron:
    def __init__(self, nb_nodes=[30], eta=1e-2, beta=1e-2, r=1e-2):
        np.seterr(over='ignore')
        np.seterr(divide='raise')
        self.w = []         # パラメータ
        self.dims = flatten([[None], nb_nodes, [None]]) # 各層の次元数
        self.nb_batch = 1   # ミニバッチの大きさ
        self.eta = eta      # 勾配法の更新率
        self.b   = beta     # シグモイド関数の傾斜パラメータ
        self.r   = r        # 正規化パラメータ
        # 各層の活性化関数
        self.s  = [self.tanh  if i < len(self.dims)-2 else self.ident  for i in range(len(self.dims)-1)]
        self.sp = [self.tanhp if i < len(self.dims)-2 else self.identp for i in range(len(self.dims)-1)]

    # シグモイド関数
    def sigm (self, x): return np.vectorize(lambda x: 1./(1+np.exp(-self.b*x)))(x)
    def sigmp(self, x): s = self.sigm(x); return self.b*s*(1-s)

    # tanh
    def tanh (self, x): return np.tanh(self.b*x)
    def tanhp(self, x): return self.b*(1-np.tanh(self.b*x)**2)

    # ReLU
    def relu (self, x): return np.vectorize(lambda x: np.log(1+np.exp(x)))(x)
    def relup(self, x): return np.vectorize(lambda x: 1./(1+np.exp(-x)))(x)

    # 恒等関数
    def ident (self, x): return x
    def identp(self, x): return 1

    def labeling(self, y):
        dim = max(y)+1
        Y = -1 * np.array([np.ones(dim) for i in range(y.shape[0])])
        for i in range(Y.shape[0]):
            Y[i, y[i]] = 1
        return Y

    def pretrain(self, x):
        # 事前学習
        for i in range(len(self.w)):
            ae = AutoEncoder(n_dim=self.dims[i+1], eta=self.eta, beta=self.b)
            ae.fit(x)
            self.w[i] = ae.weight()
            if i < len(self.w) - 1:
                # 最後の層以外は下位層の事前学習用に順伝播計算する
                x = ae.reduce(x)

    def fit(self, x, y, pretraining=False):
        # 確率的勾配降下法(SGD)で学習する
        data = zip(x, y)
        np.random.shuffle(data) # データをランダムにシャッフルする
        data = zip(*data)       # unzip
        X = np.array(data[0])
        Y = np.array(data[1])

        # 教師データのフォーマッティング
        if len(list(Y.shape)) == 1:
            Y = self.labeling(Y)

        self.dims[0]  = X.shape[1]  # 入力ベクトルの次元数
        self.dims[-1] = Y.shape[1]  # 出力ベクトルの次元数

        if pretraining:
            # 事前学習によるパラメータ初期化
            self.w = np.array([np.zeros((self.dims[i+1], self.dims[i])) for i in range(len(self.dims)-1)])
            self.pretrain(x)
        else:
            # 乱数によるパラメータ初期化
            self.w = np.array([
                    np.random.uniform(
                        -4*np.sqrt(6./(self.dims[i]+self.dims[i+1])),
                        4*np.sqrt(6./(self.dims[i]+self.dims[i+1])),
                        (self.dims[i+1], self.dims[i]))
                    for i in range(len(self.dims)-1)])

        nb_data = X.shape[0]

        for i in range(int(np.ceil(nb_data / self.nb_batch))):
            # 学習データのオンライン学習
            # 学習データ全体をミニバッチに分割して学習
            if i < np.ceil(nb_data / self.nb_batch)-1:
                self.update(X[i*self.nb_batch:(i+1)*self.nb_batch].T, Y[i*self.nb_batch:(i+1)*self.nb_batch])
            else:
                self.update(X[i*self.nb_batch:nb_data].T, Y[i*self.nb_batch:nb_data])

    def update(self, x, t):
        # 逆伝播計算
        nb_batches = 1 if len(x.shape) < 2 else x.shape[1]
        u, z = self.forward(x)
        delta = np.array([np.zeros(self.dims[i]) for i in range(len(self.w))])
        for i in range(len(self.w)-1, -1, -1):
            if i < len(self.w)-1:
                delta[i] = self.sp[i](u[i]) * np.dot(self.w[i+1].T, delta[i+1])
            else:
                delta[i] = z[-1] - t.T

        for i in range(len(self.w)):
            if i == 0:
                self.w[i] -= self.eta * (np.dot(delta[i], x.T) / nb_batches + self.r * self.w[i])
            else:
                self.w[i] -= self.eta * (np.dot(delta[i], z[i-1].T) / nb_batches + self.r * self.w[i])

    def forward(self, x):
        # 順伝播計算
        u = []
        z = []
        for i in range(len(self.w)):
            u.append(np.dot(self.w[i], z[i-1] if i > 0 else x))
            z.append(self.s[i](u[i]))

        return np.array(u), np.array(z)

    def predict(self, x):
        # 出力層の出力ベクトルからラベルを推定する
        u, z = self.forward(x)
        error = (z[-1] - np.ones(self.dims[-1]))**2
        # 最も1に近いラベルを推定値とする
        return np.argmin(error)

class AutoEncoder(Perceptron):
    def __init__(self, n_dim, eta=1e-2, beta=1e-2):
        Perceptron.__init__(self, nb_nodes=[n_dim], eta=eta, beta=beta)

    def fit(self, x):
        # 入力信号を教師信号として学習
        Perceptron.fit(self, x, x)

    def reduce(self, x):
        # 中間層出力を得る
        return self.s[0](np.dot(x, self.w[0].T))

    def weight(self):
        # 重みベクトルを返す
        return self.w[0]

if __name__ == "__main__":
    data = datasets.load_digits()
    data_x = data.data
    data_y = data.target

    train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_x, data_y, test_size=0.2)

    # Perceptronで教師データを学習する
    nn = Perceptron(nb_nodes=[48], beta=1.75e-2)
    nn.fit(train_x, train_y, pretraining=True)

    # 推定
    pred_y = [nn.predict(x) for x in test_x]
    print metrics.accuracy_score(pred_y, test_y)
