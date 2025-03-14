"""
logarithm of particle mass as coordinate (ensures non-negative values)
"""

import numpy as np


class WaterMassLogarithm:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(const, x, dm_dt):
        # x = ln(m)
        # dx_dt = 1/m(x) dm_dt
        #       = exp(-x) * dm_dt
        return np.exp(-x) * dm_dt

    @staticmethod
    def mass(x):
        return np.exp(x)

    @staticmethod
    def x(mass):
        return np.log(mass)
