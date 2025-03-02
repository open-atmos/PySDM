""" Timescale derived for tritium with assumption of zero ambient concentration - see text above
    Table 1 [Bolin 1958](https://digitallibrary.un.org/record/3892725) """

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
    def c1_coeff(
        const,
        vent_coeff_iso,
        vent_coeff,
        D_iso,
        D,
        alpha,
        rho_env_iso,
        rho_env,
        M_iso,
        pvs_iso,
        pvs_water,
        temperature,
    ):
        return RH / alpha

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau_of_rdrdt_c1(radius, r_dr_dt, c1_coeff):
        """timescale for evaporation of a falling drop with tritium"""
        return -(radius**2) / 3 / r_dr_dt / c1_coeff
