"""
logarithm of particle mass as diffusion coordinate
(ensures non-negative values)
"""

import numpy as np


class MassLogarithm:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(const, x, dm_dt):
        return dm_dt / np.exp(x)

    @staticmethod
    def mass(x):
        return np.exp(x)

    @staticmethod
    def x(mass):
        return np.log(mass)
