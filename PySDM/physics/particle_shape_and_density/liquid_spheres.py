"""
spherical particles with constant density of water
Reynolds number calculation assumes length scale is particle diameter
"""

import numpy as np


class LiquidSpheres:
    def __init__(self, _):
        pass

    @staticmethod
    def supports_mixed_phase(_=None):
        return False

    @staticmethod
    def mass_to_volume(const, mass):
        return mass / const.rho_w

    @staticmethod
    def volume_to_mass(const, volume):
        return const.rho_w * volume

    @staticmethod
    def radius_to_mass(const, radius):
        return const.rho_w * const.PI_4_3 * np.power(radius, const.THREE)

    @staticmethod
    def reynolds_number(_, radius, velocity_wrt_air, dynamic_viscosity, density):
        return 2 * radius * velocity_wrt_air * density / dynamic_viscosity
