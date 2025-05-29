"""
as in Pruppacher and Klett 2005 (eq. 13-14)
with reference to [Okuyama and Zung 1967](https://doi.org/10.1063/1.1840906)
"""

import numpy as np


class PruppacherKlett:
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rv * T)

    @staticmethod
    def D(const, D, r, lmbd):
        return D / (
            (r / (r + const.dv_pk05)) + 2 * np.sqrt(const.PI) * lmbd / r / const.MAC
        )

    @staticmethod
    def lambdaK(_, T, p):  # pylint: disable=unused-argument
        return -1

    @staticmethod
    def K(const, K, r, lmbd):  # pylint: disable=unused-argument
        return K  # TODO #1266
