import time

import numpy as np

# import matplotlib.pyplot as plt
from LIF import LIF

if __name__ == "__main__":
    time_length = 300  # 実験時間 (観測時間)
    dt = 0.5  # 時間分解能
    pre = 50  # 前ニューロンの数
    reservoir_size = 100  # リザバー層のニューロンの数
    p = 0.05

    inputs = np.zeros((pre, int(time_length / dt)))  # 入力スパイク列の初期化

    for i in inputs:
        i[:: np.random.randint(10, 100)] = 1  # 適当にスパイクを等間隔で立ててみる
        i[0] = 0  # 最初のindexは0にしておく (なんとなく気持ち悪いから)

    # input重みの初期化
    weights_in = (
        np.random.rand(reservoir_size, pre) + 20.0
    )  # 今回は前ニューロン数少なめなので大きめな重みにしておく

    # middle weight initialize
    weights_mid = np.zeros((reservoir_size, reservoir_size))
    non_zeros = np.random.rand(reservoir_size, reservoir_size)
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            if non_zeros[i][j] > p:
                weights_mid[i][j] = 0
            else:
                weights_mid[i][j] = 0.5  # reservoir weight initialize

    # neuron = LIF()
    neurons = [LIF() for _ in range(reservoir_size)]
    ResStat = np.zeros((reservoir_size, int(time_length / dt) + 1))
    ResStat[:, 0] = np.random.rand(reservoir_size)  # output

    start = time.time()

    for t in range(int(time_length / dt)):
        for i in range(reservoir_size):
            ResStat[i, t + 1] = neurons[i].calc(
                inputs[:, t], ResStat[:, t], weights_in, weights_mid[i, :], t
            )

    end = time.time()
    print(f"processing time was {end - start} sec")

"""
    # 結果の描画
    plt.figure(figsize=(12, 4))

    # 入力データ
    plt.subplot(1, 2, 1)
    t = np.arange(0, time, dt)
    spikes = [t[i == 1] for i in inputs]

    for i, s in enumerate(spikes):
        plt.scatter(s, [i for _ in range(len(s))], s=5.0, c='tab:blue')
    plt.xlim(0, time)
    plt.ylim(-1, pre)
    plt.xlabel('time [ms]')
    plt.ylabel('Neuron index')

    # 膜電位
    plt.subplot(1, 2, 2)
    plt.plot(t, v)
    plt.xlabel('time [ms]')
    plt.ylabel('Membrane potential [mV]')
    plt.show()
"""
