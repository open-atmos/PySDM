"""isotope e-fold timescale based on Fick's first law and Fourier's law"""

import numpy as np


class ZabaAndArabas2025:
    def __init__(self, _):
        pass

    @staticmethod
    def tau(mass, dm_dt):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        - rho_w denotes density of a drop"""
        return mass / dm_dt

    @staticmethod
    def isotope_dm_dt(radius, D_iso, f_m, M_ratio, b, S, R_liq, alpha, R_vap, rho_w):
        return (
            4
            * np.pi
            * radius
            * D_iso
            * f_m
            * M_ratio
            * rho_w
            * (1 + b * S)
            / (1 + S)
            * (R_liq / alpha - S * (1 + b) / (1 + b * S) * R_vap)
        )
