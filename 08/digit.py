#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics

class Perceptron:
    def __init__(self):
        np.seterr(over='ignore')
        np.seterr(divide='raise')
        self.w = []       # 中間層のパラメータ
        self.v = []       # 出力層のパラメータ
        self.in_dim = 0   # 入力層の次元数
        self.mid_dim = 30 # 中間層の次元数
        self.out_dim = 0  # 出力層の次元数
        self.eta = 0.3
        self.b   = 0.01

    def s(self, x):
        # ロジスティック関数
        return np.vectorize(lambda x: 1./(1+np.exp(-self.b*x)))(x)

    def sp(self, x):
        # ロジスティック関数の1次導関数
        s = self.s(x)
        return self.b*s*(1-s)

    def fit(self, x, y):
        # 確率的勾配降下法(SGD)で学習する
        data = zip(x, y)
        np.random.shuffle(data) # データをランダムにシャッフルする
        data = zip(*data)       # unzip
        X = np.array(data[0])
        Y = np.array(data[1])
        self.in_dim  = X.shape[1]     # 入力ベクトルの次元数
        self.out_dim = int(Y.max()+1) # 出力ベクトルの次元数
        self.w = np.random.uniform(-4*np.sqrt(6./(self.in_dim+self.mid_dim)), 4*np.sqrt(6./(self.in_dim+self.mid_dim)), (self.mid_dim, self.in_dim))
        self.v = np.random.uniform(-4*np.sqrt(6./(self.mid_dim+self.out_dim)), 4*np.sqrt(6./(self.mid_dim+self.out_dim)), (self.out_dim, self.mid_dim))
        for i in range(X.shape[0]):
            # 学習データを1つずつ学習する
            self.update(X[i], Y[i])

    def update(self, x, y):
        t = np.array([1 if y == k else -1 for k in range(self.out_dim)])
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

if __name__ == "__main__":
    nn = Perceptron()
    data = None
#    test_data = None

#    print "Loading train data..."
#    with open("train.csv", "r") as f:
#        f.readline()
#        data = np.loadtxt(f, delimiter=',')
#
#    data = data[0:200]

#    data_y = data[:, 0]
#    data_x = data[:, 1:]

    data = datasets.load_digits()
    data_x = data.data
    data_y = data.target

    train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_x, data_y, test_size=0.2)

    print "Fitting train data..."
    nn.fit(train_x, train_y)

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
