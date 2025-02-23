"""
constant rate formulation (for tests)
"""

import numpy as np


class Constant:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.J_HET)

    @staticmethod
    def j_het(const, a_w_ice):  # pylint: disable=unused-argument
        return const.J_HET
