"""
#TODO #744
"""

import numpy as np


class ConstantK:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.KERNEL_CONSTANTK_a)

    @staticmethod
    def liquid_liquid(const, v1, v2):  # pylint: disable=unused-argument
        return const.KERNEL_CONSTANTK_a
