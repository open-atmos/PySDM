"""
Equilibrium fractionation factors from [Majoube 1970](https://doi.org/10.1038/2261242a0)
(also published by the same author in [Majoube 1971](https://doi.org/10.1051/jcp/1971680625))
"""

import numpy as np


class Majoube1970:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_i_18O(const, T):
        return np.exp(
            const.MAJOUBE_1970_ALPHA_I_18O_T2 / T**2
            + const.MAJOUBE_1970_ALPHA_I_18O_T1 / T
            + const.MAJOUBE_1970_ALPHA_I_18O_T0
        )
