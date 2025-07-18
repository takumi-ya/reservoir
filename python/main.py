import time

import numpy as np
import cupy as cp
from tqdm import tqdm

# import matplotlib.pyplot as plt
from LIF import LIF

if __name__ == "__main__":
    time_length = 1  # 実験時間[ms](比較しやすいように1msに統一)
    dt = 0.1  # 時間分解能[ms](同様の理由で統一)
    nt = int(time_length / dt)  # シミュレーションステップ数
    pre = 50  # 前ニューロンの数
    reservoir_size = int(input('number of neuron: '))
    p = 0.05

    inputs = np.zeros((pre, nt))  # 入力スパイク列の初期化

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
    weights_mid = np.where(non_zeros > p, 0, 0.5)

    # neuron = LIF()
    neurons = LIF(size = reservoir_size)
    ResStat = np.zeros((reservoir_size, nt + 1))
    ResStat[:, 0] = np.random.rand(reservoir_size)  # output

    inputs = cp.array(inputs)  # 入力スパイク列をcupy配列に変換
    ResStat = cp.array(ResStat)  # 出力状態をcupy配列に変換
    weights_in = cp.array(weights_in)  # 入力重みをcupy配列に変換
    weights_mid = cp.array(weights_mid)  # 中間重みをcupy配列に変換

    start = time.perf_counter()

    for t in tqdm(range(nt)):
        ResStat[:, t + 1] = neurons.calc(
            inputs, ResStat, weights_in, weights_mid, t
        )

    ResStat = cp.asnumpy(ResStat)  # 出力状態をnumpy配列に変換

    end = time.perf_counter()
    print(
        f"processing time for 1ms simulation was {(end - start)*1000} ms when reservoir_size was {reservoir_size}"
    )

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
