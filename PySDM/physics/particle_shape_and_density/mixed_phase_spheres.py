"""
spherical particles with constant density of water or ice
"""

import numpy as np


class MixedPhaseSpheres:
    def __init__(self, _):
        pass

    @staticmethod
    def supports_mixed_phase(_=None):  # TODO: rename to "ice_phase?"
        return True

    @staticmethod
    def mass_to_volume(const, mass):
        return (
            max(const.ZERO_MASS, mass) / const.rho_w
            + min(const.ZERO_MASS, mass) / const.rho_i
        )

    @staticmethod
    def volume_to_mass(const, volume):
        return (
            np.maximum(const.ZERO_VOLUME, volume) * const.rho_w
            + np.minimum(const.ZERO_VOLUME, volume) * const.rho_i
        )

    @staticmethod
    def radius_to_mass(const, radius):
        raise NotImplementedError()

    @staticmethod
    def ice_mass_to_radius(const, ice_mass):
        return np.power(-ice_mass / const.PI_4_3 / const.rho_i, const.ONE_THIRD)
