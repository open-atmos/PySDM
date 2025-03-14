"""
spherical particles with constant density of water or ice
"""

import numpy as np


class MixedPhaseSpheres:
    def __init__(self, _):
        pass

    @staticmethod
    def supports_mixed_phase(_=None):
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
        return (
            np.maximum(const.ZERO_VOLUME, const.PI_4_3 * radius**3) * const.rho_w
            + np.minimum(const.ZERO_VOLUME, const.PI_4_3 * radius**3) * const.rho_i
        )

    @staticmethod
    def dm_dt(const, mass, r_dr_dt):
        """
        note: no ice phase support here yet! TODO #1389
        """
        return (
            4
            * const.PI
            * const.rho_w
            * np.power(mass / const.rho_w / const.PI_4_3, const.ONE_THIRD)
            * r_dr_dt
        )
