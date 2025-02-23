"""Timescale derived for tritium with assumption of zero ambient concentration - see text above
Table 1 [Bolin 1958](https://digitallibrary.un.org/record/3892725)"""

import numpy as np


class Bolin1958:  # pylint: disable=too-few-public-methods
    """Implementation of timescale used by Bolin 1958"""

    def __init__(self, const):
        assert np.isfinite(const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1)

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau(const, radius, r_dr_dt):
        """timescale for evaporation of a falling drop with tritium"""
        return (-3 / radius**2 * r_dr_dt * const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1) ** -1
