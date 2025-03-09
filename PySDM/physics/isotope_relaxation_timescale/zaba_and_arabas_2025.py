"""isotope relaxation timescale"""


class ZabaAndArabas2025:
    @staticmethod
    def tau(  # pylint: disable=too-many-arguments
        radius,
        alpha,
        D_iso,
        vent_coeff_iso,
        k_coeff_iso,
        e_s_env,
        rho_liq,
        T_env,
        Rv,
        saturation,
        R_vap_env,
        R_vap_eq,
    ):
        """alpha calcualted in the temperature of a droplet"""
        return (
            radius**2
            * alpha
            * rho_liq
            / 3
            / vent_coeff_iso
            / k_coeff_iso
            / D_iso
            * Rv
            * T_env
            / e_s_env
            / (saturation * R_vap_env / R_vap_eq - 1)
        )

    @staticmethod
    def mason_T0_to_Tinf_factor(const, vent_coeff, RH, RH_eq, Rv, T, D, pvs):
        return 1 + vent_coeff * (RH - RH_eq) / (
            const.K * Rv * T / const.lv / D / pvs + (const.lv / T / Rv - 1) / T
        )
