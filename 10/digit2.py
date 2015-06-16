#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, nb_mnodes=30, eta=0.3, beta=0.01):
        np.seterr(over='ignore')
        np.seterr(divide='raise')
        self.w = []       # 中間層のパラメータ
        self.v = []       # 出力層のパラメータ
        self.in_dim = 0   # 入力層の次元数
        self.mid_dim = nb_mnodes # 中間層の次元数
        self.out_dim = 0  # 出力層の次元数
        self.eta = eta
        self.b   = beta

    def s(self, x):
        # ロジスティック関数
        return np.vectorize(lambda x: 1./(1+np.exp(-self.b*x)))(x)

    def sp(self, x):
        # ロジスティック関数の1次導関数
        s = self.s(x)
        return self.b*s*(1-s)

    def labeling(self, y):
        dim = max(y)+1
        Y = -1 * np.array([np.ones(dim) for i in range(y.shape[0])])
        for i in range(Y.shape[0]):
            Y[i, y[i]] = 1
        return Y

    def pretrain(self, x):
        # 事前学習
        ae1 = AutoEncoder(n_dim=self.mid_dim, eta=self.eta*0.01)
        ae1.fit(x)
        self.w = ae1.weight()

        x = ae1.reduce(x)
        ae2 = AutoEncoder(n_dim=self.out_dim, eta=self.eta*0.001)
        ae2.fit(x)
        self.v = ae2.weight()

    def fit(self, x, y, pretraining=False):
        # 確率的勾配降下法(SGD)で学習する
        data = zip(x, y)
        np.random.shuffle(data) # データをランダムにシャッフルする
        data = zip(*data)       # unzip
        X = np.array(data[0])
        Y = np.array(data[1])
        if len(list(Y.shape)) == 1:
            Y = self.labeling(Y)

        self.in_dim  = X.shape[1]     # 入力ベクトルの次元数
        self.out_dim = Y.shape[1]     # 出力ベクトルの次元数

        self.w = np.random.uniform(-4*np.sqrt(6./(self.in_dim+self.mid_dim)), 4*np.sqrt(6./(self.in_dim+self.mid_dim)), (self.mid_dim, self.in_dim))
        self.v = np.random.uniform(-4*np.sqrt(6./(self.mid_dim+self.out_dim)), 4*np.sqrt(6./(self.mid_dim+self.out_dim)), (self.out_dim, self.mid_dim))

        # オートエンコーダによる事前学習
        # 第1層の重み初期値を推定する
        if pretraining:
            self.pretrain(x)

        for i in range(X.shape[0]):
            # 学習データを1つずつ学習する
            self.update(X[i], Y[i])

    def update(self, x, t):
        z, _ = self.output(x)
        self.w -= self.eta * np.array([self.sp(np.dot(self.w[j], x)) * np.dot(self.v.T[j], z - t) * x for j in range(self.mid_dim)])
        self.v -= self.eta * np.outer(z - t, self.s(np.dot(self.w, x)))

    def output(self, x):
        # 出力層と中間層を計算する
        m = self.s(np.dot(self.w, x))
        z = np.dot(self.v, m)
        return z, m

    def predict(self, x):
        # 出力層の出力ベクトルからラベルを推定する
        z, _ = self.output(x)
        error = (z - np.ones(self.out_dim))**2
        # 最も1に近いラベルを推定値とする
        return np.argmin(error)

class AutoEncoder(Perceptron):
    def __init__(self, n_dim=30, eta=0.3, beta=0.01):
        Perceptron.__init__(self, nb_mnodes=n_dim, eta=eta, beta=beta)

    def fit(self, x):
        # 入力信号を教師信号として学習
        Perceptron.fit(self, x, x)

    def reduce(self, x):
        # 中間層出力を得る
        return self.s(np.dot(x, self.w.T))

    def weight(self):
        # 重みベクトルを返す
        return self.w

if __name__ == "__main__":
    data = datasets.load_digits()
    data_x = data.data
    data_y = data.target

    train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_x, data_y, test_size=0.2)

    # Perceptronで教師データを学習する
    nn = Perceptron(nb_mnodes=16)
    nn.fit(train_x, train_y, pretraining=True)

    # 推定
    pred_y = [nn.predict(x) for x in test_x]
    print metrics.accuracy_score(pred_y, test_y)
