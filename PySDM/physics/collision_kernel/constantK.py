"""
#TODO #744
"""

import numpy as np


class ConstantK:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.KERNEL_CONSTANTK_a)

    @staticmethod
    def requested_attributes(_):
        return "", "", ""

    # pylint: disable=unused-argument
    @staticmethod
    def liquid_liquid(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        return const.KERNEL_CONSTANTK_a
