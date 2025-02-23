"""
logarithm of particle volume as coordinate (ensures non-negative values)
"""

import numpy as np


class VolumeLogarithm:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(const, x, r_dr_dt):
        return (
            np.exp(-const.TWO_THIRDS * x)
            * r_dr_dt
            * 3
            * np.power(const.PI_4_3, const.TWO_THIRDS)
        )

    @staticmethod
    def volume(x):
        return np.exp(x)

    @staticmethod
    def x(volume):
        return np.log(volume)
