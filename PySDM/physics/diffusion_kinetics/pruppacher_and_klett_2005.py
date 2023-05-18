"""
as in Pruppacher and Klett 2005 (eq. 13-14)
"""
import numpy as np


class PruppacherKlett:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rv * T)

    @staticmethod
    def D(const, D, r, lmbd, dv):
        return D / ((r / (r + dv)) + 2 * np.sqrt(const.PI) * lmbd / r / const.MAC)
