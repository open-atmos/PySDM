"""
as in [Jensen and Nugent 2017](https://doi.org/10.1175/JAS-D-15-0370.1)
(which refers to [Grabowski et al. (2011)](https://doi.org/10.1016/j.atmosres.2010.10.020)
but uses different gas constant, typo?)
"""

import numpy as np
from .grabowski_et_al_2011 import GrabowskiEtAl2011


class JensenAndNugent2017(GrabowskiEtAl2011):
    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rd * T)
