"""
single-equation approximation of the vapour and heat diffusion problem
as given in eq. 3.11 in [Mason 1971](https://archive.org/details/physicsofclouds0000maso)
(note that Mason's work did not include the ventilation factor, it
is included here for code maintainability - to reduce duplication at the
calling scopes)
"""


class Mason1971:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def r_dr_dt(const, RH_eq, T, RH, lv, pvs, D, K, ventilation_factor):
        return (
            ventilation_factor
            * (RH - RH_eq)
            / const.rho_w
            / (const.Rv * T / D / pvs + lv / K / T * (lv / T / const.Rv - 1))
        )
