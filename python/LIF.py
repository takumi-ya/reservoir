import numpy as np


class LIF:
    def __init__(
        self,
        rest: float = -65,
        ref: float = 3,
        th: float = -40,
        tc: float = 20,
        peak: float = 20,
        i: float = 0,
        tlast: float = 0,
        size: int = 100,
    ):
        """
        Leaky integrate-and-fire neuron
        :param rest: 静止膜電位 [mV]
        :param ref:  不応期 [ms]
        :param th:   発火閾値 [mV]
        :param tc:   膜時定数 [ms]
        :param peak: ピーク電位 [mV]
        """

        
        self.size = size
        self.rest = np.full(size, rest, dtype=np.float32)
        self.ref = np.full(size, ref, dtype=np.float32)
        self.th = np.full(size, th, dtype=np.float32)
        self.tc = np.full(size, tc, dtype=np.float32)
        self.peak = np.full(size, peak, dtype=np.float32)
        self.i = np.full(size, i, dtype=np.float32)  # 0           # 初期入力電流
        self.v = np.full(size, rest, dtype=np.float32)  # 初期膜電位
        self.tlast = np.full(size, tlast, dtype=np.float32)  # 最後に発火した時刻

    def calc(
        self, inputs, ResStat, weights_in, weights_mid, t, time=300, dt=0.5, tci=10
    ):
        """
        dtだけ膜電位を計算する
        スパイク1/0のみ出力データとする
        """
        # i = 0           # 初期入力電流
        # v = self.rest   # 初期膜電位
        # tlast = 0       # 最後に発火した時刻
        # monitor = []    # 膜電位の記録

        # for t in range(int(time/dt)):
        # 入力電流の計算
        # di = ((dt * t) > (tlast + self.ref)) * (-i + np.sum(inputs[:, t] * weights))
        di = ((dt * t) > (self.tlast + self.ref)) * (
            -self.i + np.sum(weights_in * inputs[:,t]) + np.sum(weights_mid * ResStat[:,t])
        )
        
        self.i += di * dt / tci

        # 膜電位の計算
        dv = ((dt * t) > (self.tlast + self.ref)) * ((-self.v + self.rest) + self.i)
        self.v += dv * dt / self.tc

        # 発火処理
        self.tlast = self.tlast + (dt * t - self.tlast) * (
            self.v >= self.th
        )  # 発火したら発火時刻を記録
        self.v = self.v + (self.peak - self.v) * (
            self.v >= self.th
        )  # 発火したら膜電位をピークへ

        # monitor.append(v >= self.th)

        self.v = self.v + (self.rest - self.v) * (
            self.v >= self.th
        )  # 発火したら静止膜電位に戻す

        # return monitor
        return self.v >= self.th
