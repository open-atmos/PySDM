"""
Equilibrium fractionation factor for Oxygen-17 from
[Barkan and Luz 2005](https://doi.org/10.1002/rcm.2250)
"""


class BarkanAndLuz2005:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_l_17O(const, _, alpha_l_18O):
        return alpha_l_18O**const.BARKAN_AND_LUZ_2005_EXPONENT
