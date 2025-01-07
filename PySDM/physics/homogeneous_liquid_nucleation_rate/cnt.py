"""
classical nucleation theory (CNT)
formulae from Seinfeld and Pandis Eq (11.47)
"""

import numpy as np


class CNT:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.sgm_w)

    @staticmethod
    def j_liq_homo(const, T, S):
        m1 = const.Mv / const.N_A  # kg per molecule
        v1 = m1 / const.rho_w  # m3 per molecule
        e_s = self.pvs_water(const, T)
        N1 = (S * e_s) / (m1 * const.Rv * T)  # molecules per m3
        return (
            ((2 * const.sgm_w) / (np.pi * m1)) ** (1 / 2)
            * (v1 * N1**2 / S)
            * np.exp(
                (-16 * np.pi * v1**2 * const.sgm_w**3)
                / (3 * const.k_B**3 * T**3 * np.log(S) ** 2)
            )
        )

    @staticmethod
    def r_liq_homo(const, T, S):
        m1 = const.Mv / const.N_A  # kg per molecule
        v1 = m1 / const.rho_w  # m3 per molecule
        return (2 * const.sgm_w * v1) / (const.k_B * T * np.log(S))
