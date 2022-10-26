"""
Formulae supporting `PySDM.dynamics.collisions.breakup_fragmentations.straub2010`
"""


import numpy as np


class Straub2010Nf:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def sigma1(const, CW):
        return np.sqrt(
            np.log((0.0125 * CW**0.5) ** 2 / 12 / const.STRAUB_E_D1**2 + 1)
        )

    @staticmethod
    def p1(const, rand, sigma1):
        return (
            const.PI
            / 6
            * np.exp(
                np.log(const.STRAUB_E_D1)
                - sigma1**2 / 2
                - sigma1
                / const.sqrt_two
                / const.sqrt_pi
                / np.log(2)
                * np.log((0.5 + rand) / (1.5 - rand))
            )
            ** 3
        )

    @staticmethod
    def p2(const, CW, rand):
        return (
            const.PI
            / 6
            * (
                const.STRAUB_MU2
                - ((0.007 * (CW - 21.0)) ** 2 / 12)
                / const.sqrt_two
                / const.sqrt_pi
                / np.log(2)
                * np.log((0.5 + rand) / (1.5 - rand))
            )
            ** 3
        )
