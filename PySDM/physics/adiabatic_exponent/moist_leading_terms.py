"""
adiabatic exponent, moist air, expanding to first order in qv, assuming qt=qv
"""


class MoistLeadingTerms:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def gamma(const, qv):
        return (
            1
            + const.Rd / const.c_vd
            + (const.Rv * qv) / (const.c_vd * (1 - qv))
            - (const.Rd * qv * const.c_vv) / (const.c_vd**2 * (1 - qv))
        )
