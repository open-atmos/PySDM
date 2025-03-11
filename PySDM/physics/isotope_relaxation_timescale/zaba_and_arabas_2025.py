"""isotope e-fold timescale based on Fick's first law and Fourier's law"""


class ZabaAndArabas2025:
    @staticmethod
    def tau(  # pylint: disable=too-many-arguments
        radius,
        alpha_env,
        D_iso,
        vent_coeff_iso,
        k_coeff_iso,
        e_env,
        e_iso_env,
        rho_liq,
        saturation,
        R_vap_env,
    ):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment"""
        return (
            radius**2
            * alpha_env
            * rho_liq
            / 3
            / vent_coeff_iso
            / k_coeff_iso
            / D_iso
            * R_vap_env
            / (e_iso_env / e_env * saturation / R_vap_env - 1)
        )

    @staticmethod
    def mason_T0_to_Tinf_factor(const, vent_coeff, RH, RH_eq, Rv, T, D, pvs):
        return 1 + vent_coeff * (RH - RH_eq) / (
            const.K * Rv * T / const.lv / D / pvs + (const.lv / T / Rv - 1) / T
        )
