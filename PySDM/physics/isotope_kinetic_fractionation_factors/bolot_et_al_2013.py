"""
kinetic fractionation factor from [Bolot et. al. 2013](https://doi.org/10.5194/acp-13-7903-2013)
eq. (1) with subsequent variable definitions on pages 7906 and 7907 and in Appendix (A6, A6).
In the paper, alpha_kinetic is probably actually alpha_effective
(contains alpha_equilibrium in numerator).
q_sat_inf in Bolot eq (A7) is substituted with equivalent form of (1/S * rho_v_inf/rho_[l,i]_inf).
"""


class BolotEtAl2013:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_kinetic(
        alpha_equilibrium,
        relative_humidity,
        heavy_to_light_diffusivity_ratio,
        heavy_to_light_ventilation_ratio,
        Ai,
    ):  # pylint: disable=too-many-positional-arguments too-many-arguments
        """
        water_vapor_density - density of water vapor at ambient temperature
        condensed_water_density - density of liquid or ice at ambient temperature
        """

        effective_supersaturation = 1 / (1 - Ai * (1 - 1 / relative_humidity))
        return effective_supersaturation / (
            alpha_equilibrium
            / heavy_to_light_diffusivity_ratio
            / heavy_to_light_ventilation_ratio
            * (effective_supersaturation - 1)
            + 1
        )

    def Ai(
        lv, cp, fv, fh, Le, R_v, T_inf, water_vapor_density, condensed_water_density
    ):  # pylint: disable=too-many-positional-arguments too-many-arguments
        return 1 / (
            1
            + lv
            / cp
            * fv
            / fh
            / Le
            / T_inf
            * (lv / R_v / T_inf - 1)
            / relative_humidity
            * water_vapor_density
            / condensed_water_density
        )
