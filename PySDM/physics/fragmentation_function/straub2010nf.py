"""
Formulae supporting `PySDM.dynamics.collisions.breakup_fragmentations.straub2010`
"""

import numpy as np


class Straub2010Nf:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def params_sigma1(const, CW):
        return np.sqrt(
            np.log(
                CW
                / 64
                / 100
                * const.CM
                * const.CM
                / 12
                / np.power(const.STRAUB_E_D1, const.TWO)
                + 1
            )
        )

    @staticmethod
    def params_mu1(const, sigma1):
        return np.log(const.STRAUB_E_D1) - np.power(sigma1, const.TWO) / 2

    @staticmethod
    def params_sigma2(const, CW):
        return max(0.0, 7 * (CW - 21) * const.CM / 1000) / np.sqrt(const.TWELVE)

    @staticmethod
    def params_mu2(const, ds):  # pylint: disable=unused-argument
        return const.STRAUB_MU2

    @staticmethod
    def params_sigma3(const, CW):
        return (1 + 0.76 * np.sqrt(CW)) * const.CM / 100 / np.sqrt(const.TWELVE)

    @staticmethod
    def params_mu3(ds):
        return 0.9 * ds
