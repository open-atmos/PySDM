"""
Equilibrium fractionation factors used in [Pierchala et al. 2022](https://10.1016/j.gca.2022.01.020)
"""

from PySDM.physics.isotope_equilibrium_fractionation_factors.barkan_and_luz_2005 import (
    BarkanAndLuz2005,
)
from PySDM.physics.isotope_equilibrium_fractionation_factors.horita_and_wesolowski_1994 import (
    HoritaAndWesolowski1994,
)


class PierchalaEtAl2022(BarkanAndLuz2005, HoritaAndWesolowski1994):
    def __init__(self, const):
        HoritaAndWesolowski1994.__init__(self, const)
        BarkanAndLuz2005.__init__(self, const)
