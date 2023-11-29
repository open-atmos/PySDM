"""
Water isotopic line excess parameters defined in
[Dansgaard 1964](https://doi.org/10.3402/tellusa.v16i4.8993)
for Northern hemisphere continental stations, except African and Near East
(weighted means) - see, e.g., abstract and Fig 10
"""


class Dansgaard1964:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def excess_d(const, delta_2H, delta_18O):
        return delta_2H - const.CRAIG_1961_SLOPE_COEFF * delta_18O

    @staticmethod
    def d18O_of_d2H(const, delta_2H):
        return (
            delta_2H - const.CRAIG_1961_INTERCEPT_COEFF
        ) / const.CRAIG_1961_SLOPE_COEFF
