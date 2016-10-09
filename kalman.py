# coding: utf-8

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

'''参考文献(1) に掲載されている matlab で書かれたカルマンフィルタを
Python と NumPy で書いた。

参考文献
(1) 足立修一, 丸田一郎, 『カルマンフィルタの基礎』, 東京電機大学出版局, 2012, p114, p115
'''


def main():
    # システム
    A, b, c = 1, 1, 1

    # 雑音(分散)
    Q, R = 1, 2

    # データ数
    N = 300

    # 雑音信号の生成
    v = npr.normal(0, Q, N)
    w = npr.normal(0, R, N)

    # 状態空間モデルを用いた時系列データの生成
    x = np.zeros(N)
    y = np.zeros(N)
    y[0] = np.array(c).T * x[0].T + w[0]
    for k in range(1, N):
        x[k] = A * x[k - 1].T + v[k - 1]
        y[k] = np.array(c).T * x[k].T + w[k]


    # カルマンフィルタによる状態推定

    # 推定値記憶領域の確保
    xhat = np.zeros(N)
    # 初期推定値
    P = 0
    xhat[0] = 0
    # 推定値の時間更新
    for k in range(1, N):
        xhat[k], P, G = kf(A, b, 0, c, Q, R, 0, y[k], xhat[k - 1], P)

    print u'観測値の平均事情誤差', sum(np.square(x - y)) / N
    print u'推定値の平均事情誤差', sum(np.square(x - xhat)) / N

    plt.plot(xrange(N), x, color='blue', label=u'x: ture value')
    plt.plot(xrange(N), y, color='green', label=u'y: observation value')
    plt.plot(xrange(N), xhat, color='red', label=u'xhat: estimation value')

    plt.legend(loc='upper left')
    plt.show()


def kf(A, B, Bu, C, Q, R, u, y, xhat, P):
    xhatm = A * xhat + Bu * u
    Pm = A * P * np.array(A).T + B * Q * np.array(B).T

    G = Pm * C / (np.array(C).T * Pm * C + R)

    xhat_new = xhatm + G * (y - np.array(C).T * xhatm)
    P_new = (np.eye(np.array(A).size) - G * np.array(C).T) * Pm

    return xhat_new, P_new, G

if __name__ == '__main__':
    main()
