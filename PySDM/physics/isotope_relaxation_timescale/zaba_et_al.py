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
    def corrected_b_coeff(rho_v, D_light, Fk):
        """
        Returns fixed version of the coefficient b
        in [Gedzelman & Arnold 1994 (J. Geophys. Res. Atmos. 99)](https://doi.org/10.1029/93JD03518)
        where water vapour density is wrongly omitted.
        """
        return rho_v * D_light * Fk

    @staticmethod
    def bolin_number(
        *,
        D_ratio_heavy_to_light,
        alpha,
        R_vap,
        R_liq,
        relative_humidity,
        b,
    ):  # pylint: disable=too-many-arguments
        """Heavy to total isotopic-timescales ratio (tau_heavy/tau_total).

        Returns:
            tau_heavy/tau_total = (relative total mass change)/(relative heavy mass change)
        """
        return (
            alpha
            * (1 - relative_humidity)
            / D_ratio_heavy_to_light
            / ((1 + b) * relative_humidity * (1 - alpha * R_vap / R_liq))
        )
