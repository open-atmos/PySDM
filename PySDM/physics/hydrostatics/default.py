"""
# TODO #407
"""

import numpy as np


class Default:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def drho_dz(
        const, g, p, T, water_vapour_mixing_ratio, lv, d_liquid_water_mixing_ratio__dz=0
    ):
        Rq = const.Rv / (1 / water_vapour_mixing_ratio + 1) + const.Rd / (
            1 + water_vapour_mixing_ratio
        )
        cp = const.c_pv / (1 / water_vapour_mixing_ratio + 1) + const.c_pd / (
            1 + water_vapour_mixing_ratio
        )
        rho = p / Rq / T
        return (
            g / T * rho * (Rq / cp - 1)
            - p * lv / cp / T**2 * d_liquid_water_mixing_ratio__dz
        ) / Rq

    # pylint: disable=too-many-arguments
    @staticmethod
    def p_of_z_assuming_const_th_and_initial_water_vapour_mixing_ratio(
        const, g, p0, thstd, water_vapour_mixing_ratio, z
    ):
        z0 = 0
        Rq = const.Rv / (1 / water_vapour_mixing_ratio + 1) + const.Rd / (
            1 + water_vapour_mixing_ratio
        )
        arg = (
            np.power(p0 / const.p1000, const.Rd_over_c_pd)
            - (z - z0) * const.Rd_over_c_pd * g / thstd / Rq
        )
        return const.p1000 * np.power(arg, 1 / const.Rd_over_c_pd)
