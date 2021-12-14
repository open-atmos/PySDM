"""
Fuch-Sutugin transition-regime correction as advocated for cloud modelling
  in [Laaksonen et al. 2005](https://doi.org/10.5194/acp-5-461-2005)
"""
from numpy import sqrt
from PySDM.physics import constants as const


class FuchsSutugin:
    @staticmethod
    def lambdaD(D, T):
        return D / sqrt(2 * const.Rv * T)

    @staticmethod
    def lambdaK(T, p):
        return (4. / 5) * const.K0 * T / p / sqrt(2 * const.Rd * T)

    @staticmethod
    def DK(DK, r, lmbd):
        return DK * (1 + lmbd/r) / (1 + 1.71 * lmbd/r + 1.33 * lmbd/r * lmbd/r)
