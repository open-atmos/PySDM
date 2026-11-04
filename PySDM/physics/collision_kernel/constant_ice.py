"""
#TODO #744
"""

import numpy as np


class ConstantIce:
    def __init__(self, const):
        assert np.isfinite(const.KERNEL_CONSTANT_ICE_a)
        assert np.isfinite(const.KERNEL_CONSTANT_ICE_b)

    @staticmethod
    def requested_attributes(_):
        return "", "", ""

    # pylint:disable=unused-argument
    @staticmethod
    def liquid_ice(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        return const.KERNEL_CONSTANT_ICE_a

    # pylint:disable=unused-argument
    @staticmethod
    def ice_ice(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        return const.KERNEL_CONSTANT_ICE_b
