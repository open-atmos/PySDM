"""
classical nucleation theory (CNT)
formulae from Seinfeld and Pandis Eq (11.47) and (11.52)
"""

import numpy as np


class CNT:
    def __init__(self, _):
        return

    @staticmethod
    def j_liq_homo(const, T, S, e_s):
        m1 = const.Mv / const.N_A  # kg per molecule
        v1 = m1 / const.rho_w  # m3 per molecule
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
