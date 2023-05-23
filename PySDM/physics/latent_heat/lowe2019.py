"""
temperature-dependent latent heat of vaporization used in Lowe et al. 2019
using equation form from Seinfeld and Pandis, with constant values from ICPM code
"""
from PySDM.physics.latent_heat.seinfeld_and_pandis_2010 import SeinfeldPandis

from ..constants import si


class Lowe2019(SeinfeldPandis):  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert const.l_l19_a == 0.167 * si.dimensionless
        assert const.l_l19_b == 3.65e-4 / si.kelvin
        SeinfeldPandis.__init__(self, const)
