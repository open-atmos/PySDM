"""
as in [Lowe et al. 2019](https://doi.org/10.1038/s41467-019-12982-0)
"""
import numpy as np


class LoweEtAl2019:
    def __init__(self, _):
        pass

    @staticmethod
    def D(const, T, p):
        return const.D0 * np.power(T / const.T0, const.D_exp) * (const.p1000 / p)

    @staticmethod
    def K(const, T, p):  # pylint: disable=unused-argument
        return 4.2e-3 * (1.0456 + 0.017 * T)
