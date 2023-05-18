"""
as in [Lowe et al. 2019](https://doi.org/10.1038/s41467-019-12982-0)
uses eq. 13-14 in Pruppacher & Klett 2005 with Delta v = 0
and no corrections for thermal conductivity
"""
from PySDM.physics.diffusion_kinetics.pruppacher_and_klett_2005 import PruppacherKlett


class LoweEtAl2019(PruppacherKlett):
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaK(const, T, p):  # pylint: disable=unused-argument
        return -1

    @staticmethod
    def K(const, K, r, lmbd):  # pylint: disable=unused-argument
        return K

    @staticmethod
    def D(const, D, r, lmbd):
        return super().D(const, D, r, lmbd, const.l19_dv)
