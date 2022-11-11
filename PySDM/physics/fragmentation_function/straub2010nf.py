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

    @staticmethod
    def p3(const, CW, ds, rand):
        return (
            const.PI
            / 6
            * (
                (0.9 * ds)
                - ((0.01 * (0.76 * CW**0.5 + 1.0)) ** 2 / 12)
                / const.sqrt_two
                / const.sqrt_pi
                / np.log(2)
                * np.log((0.5 + rand) / (1.5 - rand))
            )
            ** 3
        )

    @staticmethod
    def p4(const, CW, ds, v_max, Nr1, Nr2, Nr3):  # pylint: disable=too-many-arguments
        return (
            const.PI
            / 6
            * (
                v_max / const.PI_4_3 * 8
                + ds**3
                - Nr1
                * np.exp(
                    3 * np.log(const.STRAUB_E_D1)
                    + 6
                    * np.log(
                        (0.0125 * CW**0.5) ** 2 / 12 / const.STRAUB_E_D1**2 + 1
                    )
                    / 2
                )
                - Nr2
                * (
                    const.STRAUB_MU2**3
                    + 3 * const.STRAUB_MU2 * ((0.007 * (CW - 21.0)) ** 2 / 12) ** 2
                )
                - Nr3
                * (
                    (0.9 * ds) ** 3
                    + 3 * 0.9 * ds * ((0.01 * (0.76 * CW**0.5 + 1.0)) ** 2 / 12) ** 2
                )
            )
        )
