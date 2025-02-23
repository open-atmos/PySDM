"""
freezing temperature spectrum based on
 [Niemand et al. 2012](https://doi.org/10.1175/JAS-D-11-0249.1) INAS density parameterization
"""

import numpy as np


class Niemand_et_al_2012:
    def __str__(self):
        return "Niemand et al. 2012"

    def __init__(self, const):
        assert np.isfinite(const.NIEMAND_A)
        assert np.isfinite(const.NIEMAND_B)

    @staticmethod
    def ns(const, T):
        return np.exp(const.NIEMAND_A * (T - const.T0) + const.NIEMAND_B)

    @staticmethod
    def pdf(const, T, A_insol):
        ns_T = np.exp(const.NIEMAND_A * (T - const.T0) + const.NIEMAND_B)
        return -A_insol * const.NIEMAND_A * ns_T * np.exp(-A_insol * ns_T)

    @staticmethod
    def cdf(const, T, A_insol):
        ns_T = np.exp(const.NIEMAND_A * (T - const.T0) + const.NIEMAND_B)
        return (
            1
            - np.exp(-A_insol * ns_T)
            - np.exp(-A_insol * np.exp(-const.NIEMAND_A * const.T0 + const.NIEMAND_B))
        )

    @staticmethod
    def invcdf(const, cdf, A_insol):
        tmp = np.log(
            (
                np.log(1 - cdf)
                + np.exp(
                    -A_insol * np.exp(-const.NIEMAND_A * const.T0 + const.NIEMAND_B)
                )
            )
            / -A_insol
        )
        return const.T0 + (tmp - const.NIEMAND_B) / const.NIEMAND_A
