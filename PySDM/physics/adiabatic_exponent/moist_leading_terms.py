"""
adiabatic exponent, moist air, expanding to first order in qv, assuming qt=qv
"""


class MoistLeadingTerms:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def gamma(const, vapour_mixing_ratio):
        return (
            1
            + const.Rd / const.c_vd
            + (const.Rv * vapour_mixing_ratio)
            / (const.c_vd * (1 - vapour_mixing_ratio))
            - (const.Rd * vapour_mixing_ratio * const.c_vv)
            / (const.c_vd**2 * (1 - vapour_mixing_ratio))
        )
