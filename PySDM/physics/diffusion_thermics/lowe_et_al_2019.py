"""
as in [Lowe et al. 2019](https://doi.org/10.1038/s41467-019-12982-0)
"""

from PySDM.physics.diffusion_thermics.seinfeld_and_pandis_2010 import (
    SeinfeldAndPandis2010,
)


class LoweEtAl2019(SeinfeldAndPandis2010):
    def __init__(self, const):
        SeinfeldAndPandis2010.__init__(self, const)

    @staticmethod
    def K(const, T, p):  # pylint: disable=unused-argument
        return const.k_l19_a * (const.k_l19_b + const.k_l19_c * T)
