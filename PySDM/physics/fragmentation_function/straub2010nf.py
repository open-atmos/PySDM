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
            np.log(
                np.power((np.sqrt(CW) / 8) / 10, 2)
                / 12
                / np.power(const.STRAUB_E_D1, const.TWO)
                + 1
            )
        )

    @staticmethod
    def p1(const, rand, sigma1):
        return (
            const.PI
            / 6
            * np.power(
                np.exp(
                    np.log(const.STRAUB_E_D1)
                    - np.power(sigma1, const.TWO) / 2
                    - sigma1
                    / const.sqrt_two
                    / const.sqrt_pi
                    / const.LN_2
                    * np.log((1 / const.TWO + rand) / (const.THREE / const.TWO - rand))
                ),
                const.THREE,
            )
        )

    @staticmethod
    def p2(const, CW, rand):
        return (
            const.PI
            / 6
            * np.power(
                const.STRAUB_MU2
                - (np.power(7 * (CW - 21) / 1000, const.TWO) / 12)
                / const.sqrt_two
                / const.sqrt_pi
                / const.LN_2
                * np.log((1 / const.TWO + rand) / (const.THREE / const.TWO - rand)),
                const.THREE,
            )
        )

    @staticmethod
    def p3(const, CW, ds, rand):
        return (
            const.PI
            / 6
            * np.power(
                (9 * ds / 10)
                - (np.power((76 * np.sqrt(CW) / 100 + 1) / 100, const.TWO) / 12)
                / const.sqrt_two
                / const.sqrt_pi
                / const.LN_2
                * np.log((1 / const.TWO + rand) / (const.THREE / const.TWO - rand)),
                const.THREE,
            )
        )

    @staticmethod
    def p4(const, CW, ds, v_max, Nr1, Nr2, Nr3):  # pylint: disable=too-many-arguments
        return (
            const.PI
            / 6
            * (
                v_max / const.PI_4_3 * 8
                + np.power(ds, const.THREE)
                - Nr1
                * np.exp(
                    3 * np.log(const.STRAUB_E_D1)
                    + 6
                    * np.log(
                        np.power((np.sqrt(CW) / 8) / 10, const.TWO)
                        / 12
                        / np.power(const.STRAUB_E_D1, const.TWO)
                        + 1
                    )
                    / 2
                )
                - Nr2
                * (
                    np.power(const.STRAUB_MU2, const.THREE)
                    + 3
                    * const.STRAUB_MU2
                    * np.power(
                        np.power(7 * (CW - 21) / 1000, const.TWO) / 12, const.TWO
                    )
                )
                - Nr3
                * (
                    np.power(9 * ds / 10, const.THREE)
                    + 3
                    * 9
                    * ds
                    / 10
                    * np.power(
                        np.power((76 * np.sqrt(CW) / 100 + 1) / 100, const.TWO) / 12,
                        const.TWO,
                    )
                )
            )
        )
