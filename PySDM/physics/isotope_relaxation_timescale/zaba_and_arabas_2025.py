"""isotope e-folding timescale based on Fick's first law and Fourier's law"""

import numpy as np


class ZabaAndArabas2025:
    def __init__(self, _):
        pass

    @staticmethod
    def tau(m_dm_dt):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        """
        return 1 / m_dm_dt

    @staticmethod
    def isotope_m_dm_dt(
        const, rho_s, radius, D_iso, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
        D_iso
            Mass diffusivity coefficient for heavy isotope.
        """
        return np.abs(
            -3
            * rho_s
            / radius**2
            / const.rho_w
            / alpha
            * D_iso
            * (S * (alpha * R_vap / R_liq - 1) + (S - 1) / (1 + D * Fk))
        )
