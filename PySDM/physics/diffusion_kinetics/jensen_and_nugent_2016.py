"""
as in Jensen and Nugent 2016 (eq. 10-11)
with reference to [Grabowski et al. (2011)](https://doi:10.1016/j.atmosres.2010.10.020)
"""

import numpy as np
from .pruppacher_and_klett_2005 import PruppacherKlett


class JensenAndNugent(PruppacherKlett):  # TODO #1266: rename to Grabowski et al. 2011
    """note the use of Rd instead of Rv!"""

    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rd * T)

    @staticmethod
    def lambdaK(const, T, p):  # pylint: disable=unused-argument
        return (
            ((1.5e-11) * np.power(T, 3))
            - (4.8e-8 * np.power(T, 2))
            + (10e-4 * T)
            - (3.9 * 10e-4)
        )

    @staticmethod
    def K(const, K, r, lmbd):  # pylint: disable=unused-argument
        return const.K0
        # TODO #1266
        # return lmbd / (
        #     (r / ((0.216 * const.si.um) + r))
        #     + (
        #         (lmbd / 0.7 * r * (0.001293 * const.si.g / const.si.m**3))
        #         * (np.sqrt((2 * const.pi) / (const.Rd * T)))
        #     )
        # )
