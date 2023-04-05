"""
Formulae supporting `PySDM.dynamics.collisions.breakup_fragmentations.straub2010`
"""


import numpy as np


class Straub2010Nf:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def params_p1(const, CW):
        sigma1 = np.sqrt(
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
        mu1 = np.log(const.STRAUB_E_D1) - np.power(sigma1, const.TWO) / 2
        return (mu1, sigma1)

    @staticmethod
    def params_p2(const, CW):
        deltaD2 = 7 * (CW - 21) * const.CM / 1000
        deltaD2 = max(0.0, deltaD2)
        sigma2 = deltaD2 / np.sqrt(12)
        return (const.STRAUB_MU2, sigma2)

    @staticmethod
    def params_p3(const, CW, ds):
        mu3 = 0.9 * ds
        deltaD3 = (1 + 0.76 * np.sqrt(CW)) * const.CM / 100
        sigma3 = deltaD3 / np.sqrt(12)
        return (mu3, sigma3)

    @staticmethod
    def params_p4(vl, ds, mu1, sigma1, mu2, sigma2, mu3, sigma3, N1, N2, N3):
        # pylint: disable=too-many-arguments, too-many-locals
        M31 = N1 * np.exp(3 * mu1 + 9 * np.power(sigma1, 2) / 2)
        M32 = N2 * (mu2**3 + 3 * mu2 * sigma2**2)
        M33 = N3 * (mu3**3 + 3 * mu3 * sigma3**2)
        M34 = vl * 6 / np.pi + ds**3 - M31 - M32 - M33
        if M34 <= 0.0:
            d34 = 0
            M34 = 0
        else:
            d34 = np.exp(np.log(M34) / 3)
        return (M31, M32, M33, M34, d34)
