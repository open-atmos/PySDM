"""
classical nucleation theory (CNT)
formulae from [Seinfeld and Pandis](https://archive.org/details/0237-pdf-atmospheric-chemistry-and-physics-2nd-ed-j.-seinfeld-s.-pandis-wiley-2006-ww)
Eq (11.47) and (11.52)
"""  # pylint: disable=line-too-long

import numpy as np


class CNT:
    def __init__(self, _):
        return

    @staticmethod
    def j_liq_homo(const, T, S, e_s):
        mass_per_moleculue = const.Mv / const.N_A
        volume_per_molecule = mass_per_moleculue / const.rho_w
        N1 = (S * e_s) / (mass_per_moleculue * const.Rv * T)  # molecules per m3
        return (
            ((2 * const.sgm_w) / (np.pi * mass_per_moleculue)) ** (1 / 2)
            * (volume_per_molecule * N1**2 / S)
            * np.exp(
                (-16 * np.pi * volume_per_molecule**2 * const.sgm_w**3)
                / (3 * const.k_B**3 * T**3 * np.log(S) ** 2)
            )
        )

    @staticmethod
    def r_liq_homo(const, T, S):
        mass_per_moleculue = const.Mv / const.N_A
        volume_per_molecule = mass_per_moleculue / const.rho_w
        return (2 * const.sgm_w * volume_per_molecule) / (const.k_B * T * np.log(S))
