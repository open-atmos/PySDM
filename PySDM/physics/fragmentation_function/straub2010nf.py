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
                # np.power((np.sqrt(CW) / 8) / 10, 2)
                CW / 64 / 100 / 12 / np.power(const.STRAUB_E_D1, const.TWO)
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
    def params_p1(const, CW):
        sigma1 = np.sqrt(
            np.log(
                # np.power((np.sqrt(CW) / 8) / 10 * const.CM, 2)
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
    def params_p2(const, CW):
        mu2 = const.STRAUB_MU2
        deltaD2 = 7 * (CW - 21) / 1000 * const.CM
        deltaD2 = max(0.0, deltaD2)
        sigma2 = deltaD2 / np.sqrt(12)
        return (mu2, sigma2)

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
    def params_p3(const, CW, ds):
        mu3 = 0.9 * ds
        deltaD3 = (1 + 0.76 * np.sqrt(CW)) / 100 * const.CM
        sigma3 = deltaD3 / np.sqrt(12)
        return (mu3, sigma3)

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

    @staticmethod
    def params_p4(const, vl, ds, mu1, sigma1, mu2, sigma2, mu3, sigma3, N1, N2, N3, CW):
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

    @staticmethod
    def erfinv(X):
        return np.arctanh(2 * X - 1) * 2 * np.sqrt(3) / np.pi
