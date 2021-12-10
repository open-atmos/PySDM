import numpy as np
from PySDM.physics import constants as const

P_median = .5
DT_median = np.nan


class Bigg_1953:
    def __init__(self):
        assert np.isfinite(DT_median)

    @staticmethod
    def pdf(T, A_insol):  # pylint: disable=unused-argument
        A = np.log(1 - P_median)
        B = DT_median - const.T0
        return - A * np.exp(A * np.exp(B + T) + B + T)

    @staticmethod
    def cdf(T, A_insol):  # pylint: disable=unused-argument
        return np.exp(np.log(1 - P_median) * np.exp(DT_median - (const.T0 - T)))

    @staticmethod
    def median():
        return const.T0 - DT_median
