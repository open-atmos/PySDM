"""
basic geometric kernel (not taking fall velocity into account)
"""

import numpy as np


class SimpleGeometric:  # pylint:disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.KERNEL_SIMPLE_GEOMETRIC_c)

    @staticmethod
    def requested_attributes(_):
        return "radius", "area", ""

    # pylint:disable=unused-argument
    @staticmethod
    def liquid_liquid(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        return (
            const.KERNEL_SIMPLE_GEOMETRIC_c
            * (attr0_a + attr0_b) ** 2
            * abs(attr1_a - attr1_b)
        )
