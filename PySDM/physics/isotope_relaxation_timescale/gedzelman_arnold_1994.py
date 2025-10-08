class GedzelmanArnold1994:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def bolin_number(
        const,
        D_ratio_heavy_to_light,
        alpha,
        D_light,
        Fk_Howell,
        R_vap,
        R_liq,
        relative_humidity,  # molar_heavy
    ):  # pylint: disable=unused-argument
        # dR_liq_dt = (
        #     3
        #     * D_ratio_heavy_to_light
        #     * rho_s
        #     / const.rho_w
        #     / radius**2
        #     * (
        #         relative_humidity * (R_vap - R_liq / alpha)
        #         + (1 - relative_humidity)
        #         / (1 + Fk_Howell * D_light)
        #         * R_liq
        #         * (1 / D_ratio_heavy_to_light - 1 / alpha)
        #     )
        # )
        # dm_dt_over_m = 3 / r**2 * r_dr_dt
        # Bo = (1 / dR_liq_dt * R_liq * dm_dt_over_m)
        D_heavy = D_ratio_heavy_to_light * D_light
        RH_eq = 1  # TODO check
        Fd_Fick = const.rho_w / const.rho_STP / D_light  # TODO check
        return (
            const.rho_w
            / const.rho_STP
            * (relative_humidity - RH_eq)
            / (Fk_Howell + Fd_Fick)
            / (
                relative_humidity * D_heavy * (R_vap / R_liq - 1 / alpha)
                + (1 - relative_humidity)
                / (1 / D_heavy + Fk_Howell / D_ratio_heavy_to_light)
                * (1 / D_ratio_heavy_to_light - 1 / alpha)
            )
        )
