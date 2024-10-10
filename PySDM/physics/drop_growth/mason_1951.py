"""
single-equation approximation of the vapour and heat diffusion problem
as derived in [Mason 1951](https://doi.org/10.1088/0370-1301/64/9/307)
(note that Mason's work did not include the ventilation factor, it
is included here for code maintainability - to reduce duplication at the
calling scopes; it is not ignored and is used as a multiplicative factor
in the same way as in `PySDM.physics.drop_growth.mason_1971.Mason1971`)
"""


class Mason1951:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def r_dr_dt(const, RH_eq, T, RH, lv, pvs, D, K, ventilation_factor):
        return (
            ventilation_factor
            * (RH - RH_eq)
            / const.rho_w
            / (const.Rv * T / D / pvs + lv**2 / K / T**2 / const.Rv)
        )
