"""
saturation when no change in isotopic ratio
in liquid and in vapour
"""


# pylint: disable=too-few-public-methods
class ZabaEtAl:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def saturation_for_zero_dR_condition(
        _,
        diff_rat_heavy_to_light,
        iso_ratio_x,
        iso_ratio_r,
        iso_ratio_v,
        alpha_w,
        Fd,
        Fk,
    ):
        A = alpha_w / diff_rat_heavy_to_light * iso_ratio_x / iso_ratio_r - 1
        return A / ((1 + Fk / Fd) * (1 - alpha_w * iso_ratio_v / iso_ratio_r) + A)
