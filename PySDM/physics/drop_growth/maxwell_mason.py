"""
single-equation approximation of the vapour and heat diffusion problem
"""


class MaxwellMason:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def r_dr_dt(const, RH_eq, T, RH, lv, pvs, D, K):
        return (RH - RH_eq) / (
            const.rho_w * const.Rv * T / D / pvs
            + const.rho_w * lv / K / T * (lv / const.Rv / T - 1)
        )
