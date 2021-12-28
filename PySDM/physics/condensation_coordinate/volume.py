"""
particle volume as condensation coordinate (i.e. no transformation)
"""
import numpy as np


class Volume:
    def __init__(self, const):
        pass

    @staticmethod
    def dx_dt(const, x, r_dr_dt):
        return 4 * const.PI * np.power(x / const.PI_4_3, const.ONE_THIRD) * r_dr_dt

    @staticmethod
    def volume(const, x):
        return x

    @staticmethod
    def x(const, volume):
        return volume
