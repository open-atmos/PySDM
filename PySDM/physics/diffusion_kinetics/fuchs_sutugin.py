from PySDM.physics import constants as const
from numpy import sqrt


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
