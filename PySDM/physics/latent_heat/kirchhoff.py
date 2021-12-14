"""
[Kirchhoff's
 law](https://en.wikipedia.org/wiki/Gustav_Kirchhoff#Kirchhoff's_law_of_thermochemistry)
 based temperature-dependent latent heat of vaporization
"""
from PySDM.physics import constants as const


class Kirchhoff:
    @staticmethod
    def lv(T):
        return const.l_tri + (const.c_pv - const.c_pw) * (T - const.T_tri)
