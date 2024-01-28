"""
Equilibrium fractionation factors used in
[Graf et al. 2017](https://doi.org/10.3929/ethz-b-000266387)
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


class GrafPhD2017(Majoube1970, Majoube1971, MerlivatAndNief1967):
    def __init__(self, const):
        Majoube1970.__init__(self, const)
        Majoube1971.__init__(self, const)
        MerlivatAndNief1967.__init__(self, const)
