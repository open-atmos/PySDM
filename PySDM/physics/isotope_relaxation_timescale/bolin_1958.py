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
        return -(radius**2) / 3 / r_dr_dt * const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1

    @staticmethod
    # pylint: disable=too-many-arguments
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
        return (
            vent_coeff_iso
            * D_iso
            / vent_coeff
            / D
            / alpha
            / pvs_iso
            * pvs_water
            * (rho_env_iso / M_iso - pvs_iso / const.R_str / temperature)
            / (rho_env / const.Mv - pvs_water / const.R_str / temperature)
        )

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau_of_rdrdt_c1(const, radius, r_dr_dt, alpha, c1_coeff):
        """timescale for evaporation of a falling drop with tritium"""
        return -(radius**2) / 3 / r_dr_dt / c1_coeff

    @staticmethod
    # pylint: disable=too-many-arguments
    def tau_without_assumptions(
        const, radius, alpha, D, vent_coeff, rho_env, M, temperature, pvs_water, pvs
    ):
        return (
            alpha
            * radius**2
            / 3
            * const.rho_w
            / D
            / vent_coeff
            / const.Mv
            * pvs
            / pvs_water
            / (rho_env / M - pvs / const.R_STR / temperature)
        )
