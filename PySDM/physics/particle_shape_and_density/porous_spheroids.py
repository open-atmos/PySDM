"""
for porous spheriods  as in
[Shima et al. 2020](https://doi.org/10.5194/gmd-13-4107-2020)
and prolate spheriods as in
[Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)
"""

import numpy as np


class PorousSpheroid:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def supports_mixed_phase(_=None):
        return True

    @staticmethod
    def aspect_ratio(polar_radius, equatorial_radius):
        return polar_radius / equatorial_radius

    @staticmethod
    def equatorial_radius(polar_radius, aspect_ratio):
        return polar_radius / aspect_ratio

    @staticmethod
    def polar_radius(equatorial_radius, aspect_ratio):
        return equatorial_radius * aspect_ratio

    @staticmethod
    def eccentricity(aspect_ratio):
        return np.sqrt(1 - aspect_ratio**-2.0)
