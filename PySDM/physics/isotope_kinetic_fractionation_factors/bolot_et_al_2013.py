"""
kinetic fractionation factor from [Bolot et al. 2013](https://doi.org/10.5194/acp-13-7903-2013)
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
    def transfer_coefficient_liq_to_ice(D, Fk):
        """eq. (A6) in Bolot.

        Parameters
        ----------
        D
            light isotope diffusion coefficient
        Fk
            term associated with heat transfer

        Returns
        ----------
        A_li
            liquid to ice transfer coefficient
        """
        return 1 / (1 + D * Fk)

    @staticmethod
    def effective_supersaturation(transfer_coefficient_liq_to_ice, relative_humidity):
        return 1 / (1 - transfer_coefficient_liq_to_ice * (1 - 1 / relative_humidity))
