"""
for columnar ice crystals as in
[Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)
"""

import numpy as np
from .porous_spheroids import PorousSpheroid


class ColumnarIce(PorousSpheroid):

    @staticmethod
    def polar_radius_empirical_parametrisation(const, mass):
        """Sec. 3.1.2 in [Spichtinger & Gierens 2009].
        Note that the notation in the paper is incorrect.
        """
        return np.where(
            mass < const.columnar_ice_mass_transition,
            (mass / const.columnar_ice_length_alpha_1)
            ** (1 / const.columnar_ice_length_beta_1)
            / 2,
            (mass / const.columnar_ice_length_alpha_2)
            ** (1 / const.columnar_ice_length_beta_2)
            / 2,
        )

    @staticmethod
    def aspect_ratio_empirical_parametrisation(const, mass):
        """Eq. 17 in [Spichtinger & Gierens 2009]"""
        return np.where(
            mass < const.columnar_ice_mass_transition,
            1,
            np.sqrt(
                np.sqrt(27)
                * const.columnar_bulk_ice_density
                / 8.0
                / const.columnar_ice_length_alpha_2
                ** (3 / const.columnar_ice_length_beta_2)
            )
            * mass
            ** (
                (3 - const.columnar_ice_length_beta_2)
                / 2
                / const.columnar_ice_length_beta_2
            ),
        )
