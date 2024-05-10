"""
Equilibrium fractionation factors from
[Lamb et al. 2017](https://doi.org/10.1073/pnas.1618374114)
"""

import numpy as np


class LambEtAl2017:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_i_2H(const, T):
        return np.exp(
            const.LAMB_ET_AL_2017_ALPHA_I_2H_T2 / T**2
            + const.LAMB_ET_AL_2017_ALPHA_I_2H_T1 / T
            + const.LAMB_ET_AL_2017_ALPHA_I_2H_T0
        )
