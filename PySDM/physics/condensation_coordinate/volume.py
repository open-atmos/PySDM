from numpy import power
from PySDM.physics import constants as const


class Volume:
    @staticmethod
    def dx_dt(x, r_dr_dt):
        return 4 * const.PI * power(x / const.PI_4_3, const.ONE_THIRD) * r_dr_dt

    @staticmethod
    def volume(x):
        return x

    @staticmethod
    def x(volume):
        return volume
