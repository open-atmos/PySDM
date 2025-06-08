"""Timescale derived for tritium with assumption of zero ambient concentration - see text above
Table 1 [Bolin 1958](https://digitallibrary.un.org/record/3892725)"""

import numpy as np


class Bolin1958:  # pylint: disable=too-few-public-methods
    """Implementation of timescale used by Bolin 1958"""

    def __init__(self, const):
        assert np.isfinite(const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1)

    @staticmethod
    # pylint: disable=too-many-arguments unused-argument
    def tau_of_rdrdt(const, radius, r_dr_dt, alpha=0):
        """timescale for evaporation of a falling drop with tritium"""
        return -(radius**2) / 3 / r_dr_dt / const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1

    @staticmethod
    # pylint: disable=too-many-arguments unused-argument
    def c1_coeff(const, rho_s, R_vap):
        return R_vap * rho_s / const.rho_w
