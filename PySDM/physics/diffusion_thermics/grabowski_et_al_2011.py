"""
as in [Grabowski et al. (2011)](https://doi.org/10.1016/j.atmosres.2010.10.020)
"""


class GrabowskiEtAl2011:
    def __init__(self, _):
        pass

    @staticmethod
    def D(const, T, p):  # pylint: disable=unused-argument
        return const.diffussion_thermics_D_G11_A * (
            const.diffussion_thermics_D_G11_B * T + const.diffussion_thermics_D_G11_C
        )

    @staticmethod
    def K(const, T, p):  # pylint: disable=unused-argument
        return (
            const.diffussion_thermics_K_G11_A * T**3
            + const.diffussion_thermics_K_G11_B * T**2
            + const.diffussion_thermics_K_G11_C * T
            + const.diffussion_thermics_K_G11_D
        )
