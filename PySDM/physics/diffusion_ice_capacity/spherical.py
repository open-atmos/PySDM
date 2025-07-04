"""
capacity for approximation of ice crystals as spheres
"""

import numpy as np


class Spherical:  # pylint: disable=too-few-public-methods

    def __init__(self, _):
        pass

    @staticmethod
    def capacity(const, mass):
        return np.power(mass / const.PI_4_3 / const.rho_i, const.ONE_THIRD)
