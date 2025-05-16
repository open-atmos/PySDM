"""
Koop homogeneous nucleation rate parameterization for solution droplets
valid for 0.26 < da_w_ice < 0.34
 ([Koop et al. 2000](https://doi.org/10.1038/35020537))
"""


class Koop2000:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        pass

    @staticmethod
    def j_hom(const, T, da_w_ice):  # pylint: disable=unused-argument
        return (
            10
            ** (
                const.KOOP_2000_C1
                + const.KOOP_2000_C2 * da_w_ice
                + const.KOOP_2000_C3 * da_w_ice**2.0
                + const.KOOP_2000_C4 * da_w_ice**3.0
            )
            * const.KOOP_UNIT
        )
