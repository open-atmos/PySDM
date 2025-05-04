"""
kinetic fractionation factor from [Bolot et. al. 2013](https://doi.org/10.5194/acp-13-7903-2013)
eq. (1) with subsequent variable definitions on pages 7906 and 7907 and in Appendix (A6, A6).
In the paper, alpha_kinetic is probably actually alpha_effective
(contains alpha_equilibrium in numerator).
q_sat_inf in Bolot eq (A7) is substituted with equivalent form of (1/S * rho_v_inf/rho_[l,i]_inf).
"""

from .jouzel_and_merlivat_1984 import JouzelAndMerlivat1984


class BolotEtAl2013(JouzelAndMerlivat1984):
    def __init__(self, _):
        pass

    @staticmethod
    def transfer_coefficient_liq_to_ice(
        const,
        D,
        condensed_water_density,
        # mass_ventilation_coefficient,
        # heat_ventilation_coefficient,
        molar_mass,
        temperature,
        r_dr_dt_assuming_RHeq0_and_K_with_ventilation_coefficients,
    ):  # pylint: disable=too-many-arguments
        """
        Temperature is in 'infinity' T_inf;
        For exact formula from Bolot et al. 2013 use `r_dr_dt` from Mason 1971
            assuming `RH_eq` equal to zero and putting
            `K * heat_ventilation_coefficient / mass_ventilation_coefficient`
            instead of `K`.
        """
        return (
            r_dr_dt_assuming_RHeq0_and_K_with_ventilation_coefficients
            * condensed_water_density
            / relative_humidity
            / D
            * (const.R_str * temperature / molar_mass / const.pvs)
        )

    @staticmethod
    def effective_supersaturation(transfer_coefficient_liq_to_ice, relative_humidity):
        return 1 / (1 - transfer_coefficient_liq_to_ice * (1 - 1 / relative_humidity))
