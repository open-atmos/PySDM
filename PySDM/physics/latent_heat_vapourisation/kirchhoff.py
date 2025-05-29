"""
[Kirchhoff's
 law](https://en.wikipedia.org/wiki/Gustav_Kirchhoff#Kirchhoff's_law_of_thermochemistry)
 based temperature-dependent latent heat of vaporization
"""


class Kirchhoff:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def lv(const, T):
        return const.l_tri + (const.c_pv - const.c_pw) * (T - const.T_tri)
