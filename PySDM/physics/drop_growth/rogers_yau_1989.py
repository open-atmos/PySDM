"""
notation from [Rogers & Yau 1989](https://archive.org/details/shortcourseinclo0000roge_m3k2)
for F_k and F_d (eq. 7.17)
"""

from PySDM.physics.drop_growth import Mason1971


class RogersYau1989(Mason1971):
    def __init__(self, _):
        pass

    @staticmethod
    def Fk(const, T, K, lv):
        """thermodynamic term associated with heat conduction"""
        return const.rho_w * lv / T / K * (lv / T / const.Rv - 1)

    @staticmethod
    def Fd(const, T, D, pvs):
        """the term associated with vapour diffusion"""
        return const.rho_w * const.Rv * T / D / pvs
