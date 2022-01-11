"""
Fuch-Sutugin transition-regime correction as advocated for cloud modelling
  in [Laaksonen et al. 2005](https://doi.org/10.5194/acp-5-461-2005)
"""
import numpy as np


class FuchsSutugin:
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rv * T)

    @staticmethod
    def lambdaK(const, T, p):
        return (4. / 5) * const.K0 * T / p / np.sqrt(2 * const.Rd * T)

    @staticmethod
    def DK(_, DK, r, lmbd):
        return DK * (1 + lmbd/r) / (1 + 1.71 * lmbd/r + 1.33 * lmbd/r * lmbd/r)
