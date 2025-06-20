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
    ):
        """
        Leaky integrate-and-fire neuron
        :param rest: 静止膜電位 [mV]
        :param ref:  不応期 [ms]
        :param th:   発火閾値 [mV]
        :param tc:   膜時定数 [ms]
        :param peak: ピーク電位 [mV]
        """
        self.rest = rest
        self.ref = ref
        self.th = th
        self.tc = tc
        self.peak = peak
        self.i = i  # 0           # 初期入力電流
        self.v = rest  # 初期膜電位
        self.tlast = tlast  # 最後に発火した時刻

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
            -self.i + np.sum(inputs * weights_in) + np.sum(ResStat * weights_mid)
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
