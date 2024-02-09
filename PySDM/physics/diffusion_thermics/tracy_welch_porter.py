"""
based on "PROPERTIES OF AIR: A Manual for Use in Biophysical Ecology"
(Fourth Edition - 2010, page 22)
"""

import numpy as np


class TracyWelchPorter:
    def __init__(self, _):
        pass

    @staticmethod
    def D(const, T, p):
        return const.D0 * np.power(T / const.T0, const.D_exp) * (const.p1000 / p)

    @staticmethod
    def K(const, T, p):  # pylint: disable=unused-argument
        return const.K0
