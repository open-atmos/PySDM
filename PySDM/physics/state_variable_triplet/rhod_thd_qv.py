"""
dry-air density / dry-air potential temperature / water vapour mixing ratio triplet
(as in libcloudph++)
"""
import numpy as np


class RhodThdQv:
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
    def p(const, rhod, T, qv):
        return rhod * (1 + qv) * (const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)) * T

    @staticmethod
    def pv(const, p, qv):
        return p * qv / (qv + const.eps)

    # pylint: disable=too-many-arguments
    @staticmethod
    def dthd_dt(const, rhod, thd, T, dqv_dt, lv):
        return -lv * dqv_dt / const.c_pd / T * thd * rhod

    @staticmethod
    def th_dry(const, th_std, qv):
        return th_std * np.power(1 + qv / const.eps, const.Rd / const.c_pd)

    @staticmethod
    def rho_d(const, p, qv, theta_std):
        return (
            p
            * (1 - 1 / (1 + const.eps / qv))
            / (np.power(p / const.p1000, const.Rd_over_c_pd) * const.Rd * theta_std)
        )

    @staticmethod
    def rho_of_rhod_qv(rhod, qv):
        return rhod * (1 + qv)

    @staticmethod
    def rhod_of_pd_T(const, pd, T):
        return pd / const.Rd / T
