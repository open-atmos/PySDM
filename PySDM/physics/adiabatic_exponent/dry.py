"""
adiabatic exponent, dry air
"""


class Dry:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def gamma(const, qv):
        return 1 + const.Rd / const.c_vd
