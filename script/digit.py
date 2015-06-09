#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import cross_validation
from sklearn import metrics

class Perceptron:
    def __init__(self):
        np.seterr(over='ignore')
        np.seterr(divide='raise')
        self.w = []       # 中間層のパラメータ
        self.v = []       # 出力層のパラメータ
        self.in_dim = 0   # 入力層の次元数
        self.mid_dim = 10 # 中間層の次元数
        self.out_dim = 0  # 出力層の次元数
        self.eta = 0.1
        self.b   = 0.01
        self.data_max = 0

    def normalize(self, x):
        # 入力ベクトルの正規化
        return x/self.data_max

    def s(self, x):
        # ロジスティック関数
        return 1./(1+np.exp(-self.b*x))

    def sp(self, x):
        # ロジスティック関数の1次導関数
        s = self.s(x)
        return self.b*s/(1-s)

    def sgd(self, x, y):
        data = zip(x, y)
        np.random.shuffle(data) # データをランダムにシャッフルする
        data = zip(*data)
        X = np.array(data[0])
        Y = np.array(data[1])
        self.data_max = float(X.max())
        X = self.normalize(X)
        self.in_dim  = len(X[0])      # 入力ベクトルの次元数
        self.out_dim = int(Y.max()+1) # 出力ベクトルの次元数
        self.w = np.random.uniform(0., 1., (self.mid_dim, self.in_dim))
        self.v = np.random.uniform(0., 1., (self.out_dim, self.mid_dim))
        for i in range(len(X)):
            self.fit(X[i], Y[i])

    def fit(self, x, y):
        b = 0.01
        y = int(y)
        z, m = self.output(x)
        try:
            for j in range(self.out_dim):
                t = 1 if j == y else -1
                dprod = np.dot(self.v[j], m) 
                self.v[j] -= (self.eta * (z[j] - t) * self.sp(np.dot(self.v[j], m))) * m
                self.w -= np.array([(self.eta * (z[j] - t) * self.sp(dprod) * self.sp(np.dot(self.w[i], x)) * self.v[j][i]) * x for i in range(self.mid_dim)])
        except FloatingPointError:
            return

    def output(self, x):
        # 出力層と中間層を計算する
        x = self.normalize(x)
        m = np.array([self.s(np.dot(_w, x)) for _w in self.w])
        z = np.array([self.s(np.dot(_v, m)) for _v in self.v])
        return z, m

    def predict(self, x):
        # 出力層の出力ベクトルからラベルを推定する
        z, _ = self.output(x)
        return np.argmax(z)

if __name__ == "__main__":
    nn = Perceptron()
    data = None
#    test_data = None

    print "Loading train data..."
    with open("train.csv", "r") as f:
        f.readline()
        data = np.loadtxt(f, delimiter=',')

    data_y = data[:, 0]
    data_x = data[:, 1:]
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_x, data_y, test_size=0.2)

    print "Fitting train data..."
    nn.sgd(train_x, train_y)

    pred_y = [nn.predict(x) for x in test_x]
    print metrics.accuracy_score(pred_y, test_y)

#    print "Loading test data..."
#    with open("test.csv", "r") as f:
#        f.readline()
#        test_data = np.loadtxt(f, delimiter=',')
#
#    print "Predicting test data..."
#    pred_y = [nn.predict(x) for x in test_data]
#    with open("predict.txt", "w") as f:
#        f.write("\n".join(map(lambda x: str(x), pred_y)))
