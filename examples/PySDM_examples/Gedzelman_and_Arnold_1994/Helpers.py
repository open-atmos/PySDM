import numpy as np

from PySDM import Builder
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.physics import si
from PySDM.environments import Box
from PySDM.backends import CPU


class Commons:  # pylint: disable=too-few-public-methods
    """groups values used in both equations"""

    def __init__(self, **kwargs):
        const = kwargs["formulae"].constants
        self.vsmow_ratio = getattr(const, f'VSMOW_R_{kwargs["isotope"]}')
        self.iso_ratio_v = kwargs["formulae"].trivia.isotopic_delta_2_ratio(
            kwargs["delta_v"], self.vsmow_ratio
        )
        alpha_fun = getattr(
            kwargs["formulae"].isotope_equilibrium_fractionation_factors,
            f'alpha_l_{kwargs["isotope"]}',
        )
        if kwargs["isotope"] == "17O":
            alpha_l_18O = kwargs[
                "formulae"
            ].isotope_equilibrium_fractionation_factors.alpha_l_18O(kwargs["T"])
            self.alpha_w = alpha_fun(np.nan, alpha_l_18O)
        else:
            self.alpha_w = alpha_fun(kwargs["T"])

        self.diff_coef_ratio = 1 / getattr(
            kwargs["formulae"].isotope_diffusivity_ratios,
            f'ratio_{kwargs["isotope"]}_heavy_to_light',
        )(kwargs["T"])

        missing_b_multiplier = (
            kwargs["formulae"].saturation_vapour_pressure.pvs_water(kwargs["T"])
            / kwargs["T"]
            / const.Rv
        )
        self.b = (
            missing_b_multiplier
            * kwargs["formulae"].latent_heat_vapourisation.lv(kwargs["T"]) ** 2
            * const.D0
            / const.K0
            / const.Rv
            / kwargs["T"] ** 2
        )
        self.saturation_for_zero_dR_condition = kwargs[
            "formulae"
        ].isotope_ratio_evolution.saturation_for_zero_dR_condition
        any_number = 44.0
        self.vent_coeff_ratio = kwargs[
            "formulae"
        ].isotope_ventilation_ratio.ratio_heavy_to_light(
            ventilation_coefficient=any_number,
            diffusivity_ratio_heavy_to_light=self.diff_coef_ratio,
        )


class NoFractionationSaturation:  # pylint: disable=too-few-public-methods
    """embodies eqs. (22) an (23) from the paper"""

    def __init__(self, cmn: Commons, *, liquid: bool = False, vapour: bool = False):
        assert liquid != vapour
        self.liquid = liquid
        self.cmn = cmn

    def __call__(self, iso_ratio_r):
        return self.cmn.saturation_for_zero_dR_condition(
            iso_ratio_x=iso_ratio_r if self.liquid else self.cmn.iso_ratio_v,
            diff_rat_light_to_heavy=self.cmn.vent_coeff_ratio
            * self.cmn.diff_coef_ratio,
            b=self.cmn.b,
            alpha_w=self.cmn.alpha_w,
            iso_ratio_r=iso_ratio_r,
            iso_ratio_v=self.cmn.iso_ratio_v,
        )


class Settings:
    @staticmethod
    def make_particulator(
        *,
        formulae,
        molecular_R_liq,
        initial_R_vap=None,
        attributes=None,
        isotopes_considered=("2H",),
        n_sd=1,
        dv: float = np.nan,
        dt: float = -1 * si.s,
        RH: float = 1,
        T: float = 1,
    ):
        const = formulae.constants
        attributes["moles_2H"] = formulae.trivia.moles_heavy_atom(
            molecular_R_liq=molecular_R_liq,
            mass_total=attributes["signed water mass"],
            mass_other_heavy_isotopes=0,
            molar_mass_light_molecule=const.M_1H2_16O,
            molar_mass_heavy_molecule=const.M_2H_1H_16O,
            atoms_per_heavy_molecule=1,
        )
        builder = Builder(
            n_sd=n_sd,
            backend=CPU(
                formulae=formulae,
            ),
            environment=Box(dv=dv, dt=dt),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=isotopes_considered))

        builder.particulator.environment["RH"] = RH
        builder.particulator.environment["T"] = T
        rho_d = const.p_STP / const.Rd / T  # TODO check
        builder.particulator.environment["dry_air_density"] = rho_d

        initial_conc_vap = (
            formulae.saturation_vapour_pressure.pvs_water(T) * RH / const.R_str / T
        )
        if initial_R_vap is None:
            initial_R_vap = {}
        for isotope in HEAVY_ISOTOPES:
            initial_R_vap.setdefault(isotope, 0)
            if rho_d is not None and initial_conc_vap is not None:
                builder.particulator.environment[f"molar mixing ratio {isotope}"] = (
                    formulae.trivia.R_vap_to_molar_mixing_ratio_assuming_single_heavy_isotope(
                        R_vap=initial_R_vap[isotope],
                        density_dry_air=rho_d,
                        conc_vap_total=initial_conc_vap,
                    )
                )
            else:
                builder.particulator.environment[f"molar mixing ratio {isotope}"] = 0
        builder.request_attribute("delta_2H")
        return builder.build(attributes=attributes, products=())

    @staticmethod
    def do_one_step(formulae, particulator, evaporated_mass_fraction):
        initial_conc_vap = (
            formulae.saturation_vapour_pressure.pvs_water(
                particulator.environment["T"][0]
            )
            * particulator.environment["RH"][0]
            / formulae.constants.R_str
            / particulator.environment["T"][0]
        )
        initial_R_vap = (
            formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
                molar_mixing_ratio=particulator.environment["molar mixing ratio 2H"][0],
                density_dry_air=particulator.environment["dry_air_density"][0],
                conc_vap_total=initial_conc_vap,
            )
        )
        initial_R_liq = (
            particulator.attributes["moles_2H"][0]
            / particulator.attributes["moles_1H"][0]
        )

        dm = -evaporated_mass_fraction * (
            particulator.attributes["signed water mass"][0]
            * particulator.attributes["multiplicity"][0]
        )
        particulator.attributes["diffusional growth mass change"].data[0] = (
            dm / particulator.attributes["multiplicity"]
        )
        assert np.all(
            particulator.attributes["diffusional growth mass change"].data < 0
        )

        particulator.dynamics["IsotopicFractionation"]()

        new_R_vap = (
            formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
                molar_mixing_ratio=particulator.environment[
                    "molar mixing ratio 2H"
                ].data[0],
                density_dry_air=particulator.environment["dry_air_density"][0],
                conc_vap_total=initial_conc_vap
                - dm / formulae.constants.Mv / particulator.environment.mesh.dv,
            )
        )
        new_R_liq = (
            particulator.attributes["moles_2H"][0]
            / particulator.attributes["moles_1H"][0]
        )
        dR_vap = new_R_vap - initial_R_vap
        dR_liq = new_R_liq - initial_R_liq
        return dR_vap / initial_R_vap, dR_liq / initial_R_liq
