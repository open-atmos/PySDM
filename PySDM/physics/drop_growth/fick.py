"""
Fickian diffusion only drop growth
"""


class Fick:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def r_dr_dt(const, RH_eq, T, RH, lv, pvs, D, K):  # pylint: disable=unused-argument
        return (RH - RH_eq) / const.rho_w / (const.Rv * T / D / pvs)
