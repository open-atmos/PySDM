"""
single-equation approximation of the vapour and heat diffusion problem
as proposed in [Howell 1949](https://doi.org/10.1175/1520-0469(1949)006%3C0134:TGOCDI%3E2.0.CO;2)
same as in [Mason 1951](https://doi.org/10.1088/0370-1301/64/9/307)
"""


class Howell1949:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def r_dr_dt(const, RH_eq, T, RH, lv, pvs, D, K):
        return (
            (RH - RH_eq)
            / const.rho_w
            / (const.Rv * T / D / pvs + lv**2 / K / T**2 / const.Rv)
        )
