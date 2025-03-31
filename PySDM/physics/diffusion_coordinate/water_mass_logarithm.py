"""
logarithm of particle mass as coordinate (ensures non-negative values)
"""

import numpy as np


class WaterMassLogarithm:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(m, dm_dt):
        """
        x = ln(m/m0)
        m0 = 1 kg
        dx_dt = 1/m(x) dm_dt
        """
        return dm_dt / m

    @staticmethod
    def mass(x):
        return np.exp(x)

    @staticmethod
    def x(mass):
        return np.log(mass)

    @staticmethod
    def x_max(const):
        """corresponds to 1 kg droplet!"""
        return const.ZERO
