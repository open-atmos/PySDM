"""
Water isotopic line excess parameters defined in
[Barkan and Luz 2007](https://doi.org/10.1002/rcm.3180)
"""

import numpy as np


class BarkanAndLuz2007:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def excess_17O(const, delta_17O, delta_18O):
        return np.log(
            delta_17O + 1
        ) - const.BARKAN_AND_LUZ_2007_EXCESS_18O_COEFF * np.log(delta_18O + 1)

    @staticmethod
    def d17O_of_d18O(const, delta_18O):
        return (
            np.exp(const.BARKAN_AND_LUZ_2007_EXCESS_18O_COEFF * np.log(delta_18O + 1))
            - 1
        )
