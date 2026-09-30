"""
adiabatic exponent, dry air
"""


class Dry:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def gamma(const, qv):  # pylint: disable=unused-argument
        return 1 + const.Rd / const.c_vd
