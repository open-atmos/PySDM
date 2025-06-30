"""Timescale derived for tritium with assumption of zero ambient concentration - see text above
Table 1 [Bolin 1958](https://digitallibrary.un.org/record/3892725)"""

import numpy as np


class Bolin1958:  # pylint: disable=too-few-public-methods
    """Implementation of timescale used by Bolin 1958"""

    def __init__(self, const):
        assert np.isfinite(const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1)

    @staticmethod
    def tau(dm_dt_over_m):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        """
        return 1 / dm_dt_over_m

    @staticmethod
    def isotope_dm_dt_over_m(const, dm_dt_over_m):
        return const.BOLIN_ISOTOPE_TIMESCALE_COEFF_C1 * dm_dt_over_m

    @staticmethod
    def bolin_number(
        const,
        diffusivity_ratio_heavy_to_light,
        alpha,
        rho_s,
        Fd,
        Fk,
        saturation,
        R_vap,
        R_liq,
    ):
        """
        Bolin's number (Bo) - c1 in Bolin 1958
        """
        return (
            alpha
            / diffusivity_ratio_heavy_to_light
            / (
                (1 + const.rho_w / rho_s * Fk / Fd)
                * saturation
                * (alpha * R_vap / R_liq - 1)
                + (saturation - 1)
            )
        )
