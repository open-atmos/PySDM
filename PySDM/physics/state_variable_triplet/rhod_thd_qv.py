from PySDM.physics import constants as const
from numpy import power


class RhodThdQv:
    # A14 in libcloudph++ 1.0 paper
    @staticmethod
    def T(rhod, thd):
        return thd * (
                power(
                    (rhod * const.Rd * thd) / const.p1000 ** const.Rd_over_c_pd,
                    1 / (1 - const.Rd_over_c_pd)
                ) / const.p1000
        ) ** const.Rd_over_c_pd

    # A15 in libcloudph++ 1.0 paper
    @staticmethod
    def p(rhod, T, qv):
        return rhod * (1 + qv) * (const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)) * T

    @staticmethod
    def pv(p, qv):
        return p * qv / (qv + const.eps)

    @staticmethod
    def dthd_dt(rhod, thd, T, dqv_dt, lv):
        return - lv * dqv_dt / const.c_pd / T * thd * rhod
