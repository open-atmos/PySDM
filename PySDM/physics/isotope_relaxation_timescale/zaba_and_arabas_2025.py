"""isotope e-fold timescale based on Fick's first law and Fourier's law"""


class ZabaAndArabas2025:
    def __init__(self, _):
        pass

    @staticmethod
    def tau(  # pylint: disable=too-many-arguments
        const,
        radius,
        alpha_env,
        D_iso,
        vent_coeff_iso,
        k_coeff_iso,
        e_s,
        saturation,
        R_vap_env,
        temperature,
        M_iso,
    ):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment"""
        return (
            -(radius**2)
            * alpha_env
            * const.rho_w
            / e_s
            * const.R_str
            * temperature
            / 3
            / vent_coeff_iso
            / k_coeff_iso
            / D_iso
            * R_vap_env
            / (saturation - 1)
            / M_iso
        )

    @staticmethod
    def mason_T0_to_Tinf_factor(const, vent_coeff, RH, RH_eq, Rv, T, D, pvs):
        return 1 + vent_coeff * (RH - RH_eq) / (
            const.K * Rv * T / const.lv / D / pvs + (const.lv / T / Rv - 1) / T
        )
