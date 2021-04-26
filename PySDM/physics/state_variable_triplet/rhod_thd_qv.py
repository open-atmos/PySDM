from PySDM.physics import constants as const
from numpy import power


class RhodThdQv:
    @staticmethod
    def temperature_pressure_pv(rhod, thd, qv):
        # equivalent to eqs A11 & A12 in libcloudph++ 1.0 paper
        exponent = const.Rd / const.c_pd
        pd = power((rhod * const.Rd * thd) / const.p1000 ** exponent, 1 / (1 - exponent))
        T = thd * (pd / const.p1000) ** exponent
        R = const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)
        p = rhod * (1 + qv) * R * T
        return T, p, p - pd
