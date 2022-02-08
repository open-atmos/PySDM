"""
eq. 13-14 in Pruppacher & Klett 2005 for Delta v = 0
no corrections for thermal conductivity
"""
import numpy as np


class LoweEtAl2019:
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rv * T)

    @staticmethod
    def lambdaK(const, T, p):  # pylint: disable=unused-argument
        return -1

    @staticmethod
    def D(const, D, r, lmbd):
        return D / (1 + 2 * np.sqrt(const.PI) * lmbd / r / const.MAC)

    @staticmethod
    def K(const, K, r, lmbd):  # pylint: disable=unused-argument
        return K
