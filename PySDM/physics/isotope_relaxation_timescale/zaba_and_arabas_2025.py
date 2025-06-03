"""isotope e-fold timescale based on Fick's first law and Fourier's law"""

import numpy as np


class ZabaAndArabas2025:
    def __init__(self, _):
        pass

    @staticmethod
    def tau(const, m_dm_dt):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        """
        return 1 / m_dm_dt

    @staticmethod
    def isotope_m_dm_dt(
        const, rho_s, radius, D_iso, D, f_iso, f, S, R_liq, alpha, R_vap, Fk
    ):
        """
        Parameters
        ----------
        D_iso
            Mass diffusivity coefficient for heavy isotope.
        rho_w
            Density of liquid water.
        """
        return (
            -3
            * rho_s
            / radius**2
            / const.rho_w
            / alpha
            * D_iso
            * f_iso
            * (S * alpha * R_vap / R_liq + (S - 1) / (1 + D * f * Fk) - S)
        )
