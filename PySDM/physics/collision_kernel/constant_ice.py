"""
#TODO #744
"""

import numpy as np


class ConstantIce:
    def __init__(self, const):
        assert np.isfinite(const.KERNEL_CONSTANT_ICE_a)
        assert np.isfinite(const.KERNEL_CONSTANT_ICE_b)

    @staticmethod
    def liquid_ice(const, v1, v2):
        return const.KERNEL_CONSTANT_ICE_a

    @staticmethod
    def ice_ice(const, v1, v2):
        return const.KERNEL_CONSTANT_ICE_b
