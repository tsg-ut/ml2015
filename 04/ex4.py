# coding: utf-8
import numpy as np

# 学習率
e = np.linspace(0, 2, 100)
# 反復回数
N = 200
# 初期値ベクトル
ini = np.array([0, 0])
# 収束値ベクトル
v = np.array([0.5, 0.2])

def s(x):
    return  np.exp(-((x[0]-0.5)**2+(x[1]-0.2)**2)/2)

def s_(x):
    return np.array([0.5-x[0], 0.2-x[1]]) * s(x)

def conv(eta):
    x = ini
    for i in range(N):
        x = x + eta * s_(x)
        # 収束した時点の反復回数を返す
        if (x == v).all():
            return i
    # 収束しなかった場合
    return 1e+5 # ある程度大きい値を返す

if __name__ == '__main__':
    cnts = []
    # 各パラメータについて収束するまでの反復回数を調べる
    for eta in e:
        cnts.append(conv(eta))
    # 最も反復回数の少なかったパラメータを調べる
    i = np.argmin(cnts)
    print("eta={} => {}".format(e[i], cnts[i]))
