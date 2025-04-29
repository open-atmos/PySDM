"""
kinetic fractionation factor from [Bolot et. al. 2013](https://doi.org/10.5194/acp-13-7903-2013)
eq. (1) with subsequent variable definitions on pages 7906 and 7907 and in Appendix (A6, A6).
In the paper, alpha_kinetic is probably actually alpha_effective
(contains alpha_equilibrium in numerator).
q_sat_inf in Bolot eq (A7) is substituted with equivalent form of (1/S * rho_v_inf/rho_[l,i]_inf).
"""

from PySDM.physics.isotope_kinetic_fractionation_factors import JouzelAndMerlivat1984


class BolotEtAl2013(JouzelAndMerlivat1984):
    def __init__(self, _):
        pass

    @staticmethod
    def transfer_coefficient_liq_to_ice(
        const,
        lv,
        mass_ventilation_coefficient,
        heat_ventilation_coefficient,
        Lewis,
        molar_mass,
        temperature,
        water_vapor_density,
        condensed_water_density,
        r_dr_dt_assumid_bla,
    ):  # pylint: disable=too-many-arguments
        """
        temperature in 'infinity' T_inf
        """
        return 1 / (
            1
            + lv
            / const.c_pv
            * mass_ventilation_coefficient
            / heat_ventilation_coefficient
            / Lewis
            / temperature
            # * (lv / const.R_str / molar_mass / temperature - 1)
            # / relative_humidity
            # * water_vapor_density
            # / condensed_water_density
        )

    @staticmethod
    def effective_supersaturation(transfer_coefficient_liq_to_ice, relative_humidity):
        return 1 / (1 - transfer_coefficient_liq_to_ice * (1 - 1 / relative_humidity))
