from numpy import power
from PySDM.physics import constants as const


class Volume:
    @staticmethod
    def dx_dt(x, r_dr_dt):
        return 4 * const.pi * power(x / const.pi_4_3, const.one_third) * r_dr_dt

    @staticmethod
    def volume(x):
        return x

    @staticmethod
    def x(volume):
        return volume
