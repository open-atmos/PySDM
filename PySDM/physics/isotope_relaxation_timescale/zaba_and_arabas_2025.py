"""isotope e-folding timescale based on Fick's first law and Fourier's law"""

import numpy as np


class ZabaAndArabas2025:
    def __init__(self, _):
        pass

    @staticmethod
    def tau(dm_dt_over_m):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        """
        return 1 / dm_dt_over_m

    @staticmethod
    def isotope_dm_dt_over_m(
        *, const, rho_s, radius, D_ratio_heavy_to_light, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=too-many-arguments
        return (
            3
            * rho_s
            / radius**2
            / const.rho_w
            / alpha
            * D_ratio_heavy_to_light
            * D
            * (S * (alpha * R_vap / R_liq - 1) + (S - 1) / (1 + D * Fk))
        )
