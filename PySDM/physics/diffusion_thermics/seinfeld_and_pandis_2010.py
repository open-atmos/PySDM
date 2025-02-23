"""
as in Seinfeld and Pandis 2010 (eq. 15.65)
"""

import numpy as np


class SeinfeldAndPandis2010:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def D(const, T, p):
        return const.d_l19_a * (const.p_STP / p) * np.power(T / const.T0, const.d_l19_b)
