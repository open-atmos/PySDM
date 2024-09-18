"""
particle mass as diffusion coordinate (i.e. no transformation)
"""

import numpy as np


class Mass:
    def __init__(self, _):
        pass

    @staticmethod
    def dx_dt(const, x, dm_dt):
        return dm_dt

    @staticmethod
    def mass(_, x):
        return x

    @staticmethod
    def x(_, mass):
        return mass
