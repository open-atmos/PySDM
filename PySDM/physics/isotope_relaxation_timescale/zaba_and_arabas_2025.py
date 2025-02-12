"""isotope relaxation timescale"""


class ZabaAndArabas2025:
    @staticmethod
    def tau(
        const,
        radius,
        alpha,
        D_iso,
        vent_coeff_iso,
        rho_env,
        M_iso,
        temperature,
        Rv,
        pvs_iso,
    ):
        return (
            alpha
            * radius**2
            / 3
            * const.rho_w
            / D_iso
            / vent_coeff_iso
            / const.Mv
            * Rv
            / (rho_env / M_iso - pvs_iso / const.R_STR / temperature)
        )
