"""
dry-air density / dry-air potential temperature / water vapour mixing ratio triplet
(as in libcloudph++)
"""

import numpy as np


class LibcloudphPlusPlus:
    def __init__(self, _):
        pass

    # A14 in libcloudph++ 1.0 paper
    @staticmethod
    def T(const, rhod, thd):
        return thd * np.power(
            rhod * thd / const.p1000 * const.Rd,
            const.Rd_over_c_pd / (1 - const.Rd_over_c_pd),
        )

    # A15 in libcloudph++ 1.0 paper
    @staticmethod
    def p(const, rhod, T, water_vapour_mixing_ratio):
        return (
            rhod
            * (1 + water_vapour_mixing_ratio)
            * (
                const.Rv / (1 / water_vapour_mixing_ratio + 1)
                + const.Rd / (1 + water_vapour_mixing_ratio)
            )
            * T
        )

    @staticmethod
    def pv(const, p, water_vapour_mixing_ratio):
        return p * water_vapour_mixing_ratio / (water_vapour_mixing_ratio + const.eps)

    # pylint: disable=too-many-arguments
    @staticmethod
    def dthd_dt(const, rhod, thd, T, d_water_vapour_mixing_ratio__dt, lv):
        return -lv * d_water_vapour_mixing_ratio__dt / const.c_pd / T * thd * rhod

    @staticmethod
    def th_dry(const, th_std, water_vapour_mixing_ratio):
        return th_std * np.power(
            1 + water_vapour_mixing_ratio / const.eps, const.Rd / const.c_pd
        )

    @staticmethod
    def rho_d(const, p, water_vapour_mixing_ratio, theta_std):
        return (
            p
            * (1 - 1 / (1 + const.eps / water_vapour_mixing_ratio))
            / (np.power(p / const.p1000, const.Rd_over_c_pd) * const.Rd * theta_std)
        )

    @staticmethod
    def rho_of_rhod_and_water_vapour_mixing_ratio(rhod, water_vapour_mixing_ratio):
        return rhod * (1 + water_vapour_mixing_ratio)

    @staticmethod
    def rhod_of_pd_T(const, pd, T):
        return pd / const.Rd / T
