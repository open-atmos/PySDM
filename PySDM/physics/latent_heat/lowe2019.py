"""
temperature-dependent latent heat of vaporization used in Lowe et al. 2019
"""

class Lowe2019:
    def __init__(self, _):
        pass

    @staticmethod
    def lv(const, T):
        return const.l_tri * (const.T_tri / T)^(0.167 + 3.65e-4 * T)
