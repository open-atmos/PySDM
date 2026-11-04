"""
piecewise kernel from Eq (11) in
[Long 1974](https://doi.org/10.1175/1520-0469%281974%29031%3C1040%3ASTTDCE%3E2.0.CO%3B2)
"""

import numpy as np


class Long1974:  # pylint:disable=too-few-public-methods
    @staticmethod
    def requested_attributes(_):
        return "volume", "radius", ""

    def __init__(self, _):
        pass

    # pylint:disable=unused-argument
    @staticmethod
    def liquid_liquid(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        v_ratio = (attr0_a + attr0_b) * const.KERNEL_LONG_lin_coeff
        return np.where(
            max(attr1_a, attr1_b) < const.KERNEL_LONG_r_thres,
            (v_ratio**2 + 1) * max(attr0_a, attr0_b) ** 2 * const.KERNEL_LONG_sq_coeff,
            v_ratio,
        )
