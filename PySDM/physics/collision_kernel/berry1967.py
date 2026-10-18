"""
kernel modelling influence of electric field of 3000V/cm
 as in [Berry 1967](https://doi.org/10.1175/1520-0469(1967)024%3C0688:CDGBC%3E2.0.CO;2)
"""

import numpy as np


class Berry1967:  # pylint:disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.KERNEL_BERRY_A)
        assert np.isfinite(const.KERNEL_BERRY_B)
        assert np.isfinite(const.KERNEL_BERRY_D1)
        assert np.isfinite(const.KERNEL_BERRY_D2)
        assert np.isfinite(const.KERNEL_BERRY_E1)
        assert np.isfinite(const.KERNEL_BERRY_E2)
        assert np.isfinite(const.KERNEL_BERRY_F1)
        assert np.isfinite(const.KERNEL_BERRY_F2)
        assert np.isfinite(const.KERNEL_BERRY_G1)
        assert np.isfinite(const.KERNEL_BERRY_G2)
        assert np.isfinite(const.KERNEL_BERRY_G3)
        assert np.isfinite(const.KERNEL_BERRY_Mf)
        assert np.isfinite(const.KERNEL_BERRY_Mg)
        assert np.isfinite(const.KERNEL_BERRY_unit)

    @staticmethod
    def requested_attributes(_):
        return "radius", "relative fall velocity", ""

    @staticmethod
    def liquid_liquid(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        output = 0
        if attr0_a > attr0_b:
            r = attr0_a / const.KERNEL_BERRY_unit
            r_s = attr0_b / const.KERNEL_BERRY_unit
        else:
            r = attr0_b / const.KERNEL_BERRY_unit
            r_s = attr0_a / const.KERNEL_BERRY_unit
        p = r_s / r
        if p not in (0, 1):
            G = (
                (const.KERNEL_BERRY_G1 / r) ** const.KERNEL_BERRY_Mg
                + const.KERNEL_BERRY_G2
                + const.KERNEL_BERRY_G3 * r
            )
            Gp = (1 - p) ** G
            if Gp != 0:
                D = const.KERNEL_BERRY_D1 / r**const.KERNEL_BERRY_D2
                E = const.KERNEL_BERRY_E1 / r**const.KERNEL_BERRY_E2
                F = (
                    const.KERNEL_BERRY_F1 / r
                ) ** const.KERNEL_BERRY_Mf + const.KERNEL_BERRY_F2
                output = (
                    const.PI
                    * max(
                        0,
                        const.KERNEL_BERRY_A
                        + const.KERNEL_BERRY_B * p
                        + D / p**F
                        + E / Gp,
                    )
                    ** 2
                    * max(attr0_a, attr0_b) ** 2
                    * abs(attr1_a - attr1_b)
                )
        return output
