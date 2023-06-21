"""
Water isotopic line excess parameters defined in
[Dansgaard 1964](https://doi.org/10.3402/tellusa.v16i4.8993)
"""


class Dansgaard1964:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def excess_d(const, delta_2H, delta_18O):
        return delta_2H - const.CRAIG_1961_SLOPE_COEFF * delta_18O
