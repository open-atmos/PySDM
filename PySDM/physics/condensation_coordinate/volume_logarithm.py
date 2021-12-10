from numpy import log, exp, power
from PySDM.physics import constants as const


class VolumeLogarithm:
    @staticmethod
    def dx_dt(x, r_dr_dt):
        return exp(-const.TWO_THIRDS * x) * r_dr_dt * 3 * power(const.PI_4_3, const.TWO_THIRDS)

    @staticmethod
    def volume(x):
        return exp(x)

    @staticmethod
    def x(volume):
        return log(volume)
