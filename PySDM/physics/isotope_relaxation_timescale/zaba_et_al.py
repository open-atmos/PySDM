"""isotope e-folding timescale based on Fick's first law and Fourier's law"""


class ZabaEtAl:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(
        const, rho_s, radius, D_iso, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=too-many-arguments, unused-argument
        """relative growth of heavy isotope as a function of mass"""
        return 1 / (
            3
            * rho_s
            / radius**2
            / const.rho_w
            / alpha
            * D_iso
            * (S * (alpha * R_vap / R_liq - 1) + (S - 1) / (1 + D * Fk))
        )

    @staticmethod
    def bolin_number(
        const,
        diffusivity_ratio_heavy_to_light,
        alpha,
        rho_s,
        Fd,
        Fk,
        saturation,
        R_vap,
        R_liq,
    ):  # pylint: disable=too-many-arguments
        """
        Bolin number (Bo) - c1 in Bolin 1958
        """
        return (
            alpha
            / diffusivity_ratio_heavy_to_light
            / (
                (1 + const.rho_w / rho_s * Fk / Fd)
                * saturation
                * (alpha * R_vap / R_liq - 1)
                + (saturation - 1)
            )
        )
