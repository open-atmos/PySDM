"""
temperature-dependent latent heat of vaporization used in Lowe et al. 2019
using equation form from Seinfeld and Pandis, with constant values from ICPM code
"""

from PySDM.physics import constants_defaults
from PySDM.physics.latent_heat.seinfeld_and_pandis_2010 import SeinfeldPandis


class Lowe2019(SeinfeldPandis):  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert const.l_l19_a == constants_defaults.l_l19_a
        assert const.l_l19_b == constants_defaults.l_l19_b
        SeinfeldPandis.__init__(self, const)
