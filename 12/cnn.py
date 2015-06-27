#!/usr/bin/env python
# coding: utf-8

class ConvolutionNeuralNetwork:
    def __init__(self, nb_layer=2):
        pass

    def fit(self, x, y):
        # 確率的勾配降下法(SGD)で学習する
        data = zip(x, y)
        np.random.shuffle(data) # データをランダムにシャッフルする
        data = zip(*data)       # unzip
        X = np.array(data[0])
        Y = np.array(data[1])

        for i in range(X.shape[0]):
            # 学習データを1つずつ学習する
            self.update(X[i], Y[i])

    def update(self, x, y):
        pass

    def predict(self, x):
        pass

if __name__ == "__main__":
    nn = ConvolutionNeuralNetwork()
