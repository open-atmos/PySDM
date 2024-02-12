"""
as in Jensen and Nugent 2016 (eq. 10-11)
with reference to [Grabowski et al. (2011)](https://doi:10.1016/j.atmosres.2010.10.020)
"""

import numpy as np


class JensenAndNugent:
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return 10e-5 * ((0.15 * T) - 1.9)

    @staticmethod
    def D(const, D, r, lmbd, T):
        return lmbd / (
            (r / ((0.104 * const.si.um) + r))
            + ((lmbd / 0.036 * r) * (np.sqrt((2 * const.pi) / (const.Rd * T))))
        )
