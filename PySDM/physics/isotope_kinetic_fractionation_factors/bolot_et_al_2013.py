"""
kinetic fractionation factor from [Bolot et. al. 2013](https://doi.org/10.5194/acp-13-7903-2013)
eq. (1) with subsequent variable definitions on pages 7906 and 7907 and in Appendix (A6, A6).
In the paper, alpha_kinetic is probably actually alpha_effective
(contains alpha_equilibrium in numerator).
q_sat_inf in Bolot eq (A7) is substituted with equivalent form of (1/S * rho_v_inf/rho_[l,i]_inf).
"""


class BolotEtAl2013:
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_kinetic(
        alpha_equilibrium,
        heavy_to_light_diffusivity_ratio,
        heavy_to_light_ventilation_ratio,
        effective_supersaturation,
    ):
        """
        water_vapor_density - density of water vapor at ambient temperature
        condensed_water_density - density of liquid or ice at ambient temperature
        """
        return effective_supersaturation / (
            alpha_equilibrium
            / heavy_to_light_diffusivity_ratio
            / heavy_to_light_ventilation_ratio
            * (effective_supersaturation - 1)
            + 1
        )

    @staticmethod
    def transfer_coefficient_liq_to_ice(
        const,
        lv,
        ventilation_coefficient,
        fh,
        Lewis,
        molar_mass,
        temperature,
        water_vapor_density,
        condensed_water_density,
        relative_humidity,
    ):  # pylint: disable=too-many-arguments
        """
        temperature in 'infinity' T_inf
        """
        return 1 / (
            1
            + lv
            / const.c_pv
            * ventilation_coefficient
            / fh
            / Lewis
            / temperature
            * (lv / const.R_str / molar_mass / temperature - 1)
            / relative_humidity
            * water_vapor_density
            / condensed_water_density
        )

    @staticmethod
    def effective_supersaturation(transfer_coefficient_liq_to_ice, relative_humidity):
        return 1 / (1 - transfer_coefficient_liq_to_ice * (1 - 1 / relative_humidity))
