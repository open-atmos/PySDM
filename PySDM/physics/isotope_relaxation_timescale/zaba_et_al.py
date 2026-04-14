"""isotope e-folding timescale based on Fick's first law and Fourier's law"""


class ZabaEtAl:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(
        const, *, rho_s, radius, D_iso, D, S, R_liq, alpha, R_vap, Fk
    ):  # pylint: disable=unused-argument
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
        D_ratio_heavy_to_light,
        alpha,
        D_light,
        Fk,
        R_vap,
        R_liq,
        relative_humidity,
        rho_v,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        # TODO #1809 Numba can't compile when * in bolin_number
        """Heavy to total isotopic-timescales ratio (tau_heavy/tau_total).

        Returns:
            tau_heavy/tau_total = (relative total mass change)/(relative heavy mass change)
        """
        return (
            alpha
            * (1 - relative_humidity)
            / D_ratio_heavy_to_light
            / (
                (1 + rho_v * D_light * Fk)
                * relative_humidity
                * (1 - alpha * R_vap / R_liq)
            )
        )
