import PySDM.physics.constants as const


class MaxwellMason:
    @staticmethod
    def dr_dt(r, RH_eq, T, RH, lv, pvs, D, K):
        return (RH - RH_eq) / (
                const.rho_w * const.Rv * T / D / pvs +
                const.rho_w * lv / K / T * (lv / const.Rv / T - 1)
        ) / r
