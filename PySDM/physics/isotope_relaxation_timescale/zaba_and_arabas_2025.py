"""isotope e-fold timescale based on Fick's first law and Fourier's law"""


class ZabaAndArabas2025:
    def __init__(self, _):
        pass

    @staticmethod
    def tau(  # pylint: disable=too-many-arguments
        *,
        const,
        radius,
        Rv,
        R_liq,
        temperature,
        D_isotope,
        f_isotope,
        k_isotope,
        Rv_env,
        RH
    ):
        """e-fold timescale with alpha and water vapour pressures heavy and light water
        calculated in the temperature of environment:
        - rho_w denotes density of a drop"""
        return (
            -(radius**2)
            * const.rho_w
            / 3
            / D_isotope
            / f_isotope
            / k_isotope
            / (const.pvs_water / const.R_str / temperature * const.Mv)
            * R_liq
            / (Rv_env * RH - Rv)
        )

    @staticmethod
    def mason_T0_to_Tinf_factor(const, vent_coeff, RH, RH_eq, Rv, T, D, pvs):
        return 1 + vent_coeff * (RH - RH_eq) / (
            const.K * Rv * T / const.lv / D / pvs + (const.lv / T / Rv - 1) / T
        )
