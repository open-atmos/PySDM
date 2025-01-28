""" Timescale derived for tritium with assumption of zero ambient concentration - see text above
    Table 1 [Bolin 1958](https://digitallibrary.un.org/record/3892725) """

import numpy as np


class Bolin1958:  # pylint: disable=too-few-public-methods
    """Implementation of timescale used by Bolin 1958"""

    def __init__(self, const):
        assert np.isfinite(const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1)

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau(const, radius, r_dr_dt, alpha):
        """timescale for evaporation of a falling drop with tritium"""
        return -(radius**2) / 3 / r_dr_dt / const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1

    @staticmethod
    def tau_of_rdrdt(const, radius, r_dr_dt, alpha, R_liq, rho_v):
        """timescale for evaporation of a falling drop with tritium"""
        bolin_coeff_c1 = R_liq * const.rho_w / alpha / rho_v
        return -(radius**2) / 3 / r_dr_dt / bolin_coeff_c1

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau_without_assumptions(
        const, radius, alpha, rho_v, D, vent_coeff, Mv, rho_env_iso, Mv_iso, Rv_iso
    ):
        return (
            alpha
            * radius**2
            * const.rho_w
            / 3
            / D
            / vent_coeff
            / (rho_v - Mv * rho_env_iso / Mv_iso / Rv_iso)
        )
