"""
[Kirchhoff's
 law](https://en.wikipedia.org/wiki/Gustav_Kirchhoff#Kirchhoff's_law_of_thermochemistry)
 based temperature-dependent latent heat of vaporization
"""


class Kirchhoff:
    def __init__(self, _):
        pass

    @staticmethod
    def lv(const, T):
        return const.l_tri * (const.T_tri / T)^(0.167 + 3.65e-4 * T)
        