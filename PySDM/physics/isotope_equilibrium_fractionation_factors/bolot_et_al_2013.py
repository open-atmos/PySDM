"""
Equilibrium fractionation factors used in [Bolot et al. 2013](https://10.5194/acp-13-7903-2013)
"""

from PySDM.physics.isotope_equilibrium_fractionation_factors.majoube_1970 import (
    Majoube1970,
)
from PySDM.physics.isotope_equilibrium_fractionation_factors.majoube_1971 import (
    Majoube1971,
)
from PySDM.physics.isotope_equilibrium_fractionation_factors.merlivat_and_nief_1967 import (
    MerlivatAndNief1967,
)


class BolotEtAl2013(MerlivatAndNief1967, Majoube1970, Majoube1971):
    def __init__(self, const):
        MerlivatAndNief1967.__init__(self, const)
        Majoube1970.__init__(self, const)
        Majoube1971.__init__(self, const)
