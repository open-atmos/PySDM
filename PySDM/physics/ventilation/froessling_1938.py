"""
based on Froessling ,N. (1938) Beitr. Geophys. 52 pp. 170-216
as referenced in [Squires 1952](https://doi.org/10.1071/CH9520059)
"""


class Froessling1938:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def ventilation_coefficient(const, sqrt_re_times_cbrt_sc):
        return const.FROESSLING_1938_A + const.FROESSLING_1938_B * sqrt_re_times_cbrt_sc
