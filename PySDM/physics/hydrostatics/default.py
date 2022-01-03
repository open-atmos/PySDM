"""
# TODO #407
"""
import numpy as np


class Default:
    def __init__(self, _):
        pass

    @staticmethod
    def drho_dz(const, g, p, T, qv, lv, dql_dz=0):
        Rq = const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)
        cp = const.c_pv / (1 / qv + 1) + const.c_pd / (1 + qv)
        rho = p / Rq / T
        return (g / T * rho * (Rq / cp - 1) - p * lv / cp / T**2 * dql_dz) / Rq

    @staticmethod
    def p_of_z_assuming_const_th_and_qv(const, g, p0, thstd, qv, z):
        z0 = 0
        Rq = const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)
        arg = np.power(
            p0/const.p1000,
            const.Rd_over_c_pd
        ) - (z-z0) * const.Rd_over_c_pd * g / thstd / Rq
        return const.p1000 * np.power(arg, 1/const.Rd_over_c_pd)
