"""
assuming constant temperature and variable
gravitational acceleration g(z) = g0 * R^2 / (R+z)^2
as in [Toon et al. 1980](https://doi.org/10.1016/0019-1035(80)90173-6)
"""

import numpy as np
from PySDM.physics import constants_defaults


class VariableGIsothermal:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.celestial_body_radius)
        assert const.g_std != constants_defaults.g_std

    @staticmethod
    def pressure(const, z, p0, temperature, molar_mass):
        return p0 * np.exp(
            -const.g_std
            / const.R_str
            * molar_mass
            / temperature
            * z
            / (1 + z / const.celestial_body_radius)
        )
