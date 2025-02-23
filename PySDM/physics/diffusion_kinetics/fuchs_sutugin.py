"""
Fuch-Sutugin transition-regime correction as advocated for cloud modelling
  in [Laaksonen et al. 2005](https://doi.org/10.5194/acp-5-461-2005)
"""

import numpy as np


class FuchsSutugin:
    def __init__(self, _):
        pass

    @staticmethod
    def lambdaD(const, D, T):
        return D / np.sqrt(2 * const.Rv * T)

    @staticmethod
    def lambdaK(const, T, p):
        return (4.0 / 5) * const.K0 * T / p / np.sqrt(2 * const.Rd * T)

    @staticmethod
    def D(const, D, r, lmbd):
        return (
            D
            * (1 + lmbd / r)
            / (
                1
                + (4.0 / 3 / const.MAC + 0.377) * lmbd / r
                + (4.0 / 3 / const.MAC) * lmbd / r * lmbd / r
            )
        )

    @staticmethod
    def K(const, K, r, lmbd):
        return (
            K
            * (1 + lmbd / r)
            / (
                1
                + (4.0 / 3 / const.HAC + 0.377) * lmbd / r
                + (4.0 / 3 / const.HAC) * lmbd / r * lmbd / r
            )
        )
