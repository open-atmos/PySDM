"""isotope relaxation timescale"""


class ZabaAndArabas2025:
    @staticmethod
    def tau(
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
