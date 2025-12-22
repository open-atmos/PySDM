"""
unit test for the IsotopicFractionation dynamic
"""

from contextlib import nullcontext

import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES, LIGHT_ISOTOPES
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.physics.constants_defaults import VSMOW_R_2H  # TODO!

#
# def make_particulator(
#     *,
#     formulae,
#     backend_class,
#     molecular_R_liq,
#     initial_R_vap=None,
#     attributes=None,
#     isotopes_considered=("2H",),
#     n_sd=1,
#     dv: float = np.nan,
#     dt: float = -1 * si.s,
#     RH: float = 1,
#     T: float = 1,
# ):
#     const = formulae.constants
#     attributes["moles_2H"] = formulae.trivia.moles_heavy_atom(
#         molecular_R_liq=molecular_R_liq,
#         mass_total=attributes["signed water mass"],
#         mass_other_heavy_isotopes=0,
#         molar_mass_light_molecule=const.M_1H2_16O,
#         molar_mass_heavy_molecule=const.M_2H_1H_16O,
#     )
#     builder = Builder(
#         n_sd=n_sd,
#         backend=backend_class(
#             formulae=formulae,
#         ),
#         environment=Box(dv=dv, dt=dt),
#     )
#     builder.add_dynamic(Condensation())
#     builder.add_dynamic(IsotopicFractionation(isotopes=isotopes_considered))
#
#     builder.particulator.environment["RH"] = RH
#     builder.particulator.environment["T"] = T
#     rho_d = const.p_STP / const.Rd / T  # TODO check
#     builder.particulator.environment["dry_air_density"] = rho_d
#
#     initial_conc_vap = (
#         formulae.saturation_vapour_pressure.pvs_water(T) * RH / const.R_str / T
#     )
#     if initial_R_vap is None:
#         initial_R_vap = {}
#     for isotope in HEAVY_ISOTOPES:
#         initial_R_vap.setdefault(isotope, 0)
#         if rho_d is not None and initial_conc_vap is not None:
#             builder.particulator.environment[f"molar mixing ratio {isotope}"] = (
#                 formulae.trivia.R_vap_to_molar_mixing_ratio_assuming_single_heavy_isotope(
#                     R_vap=initial_R_vap[isotope],
#                     density_dry_air=rho_d,
#                     conc_vap_total=initial_conc_vap,
#                 )
#             )
#         else:
#             builder.particulator.environment[f"molar mixing ratio {isotope}"] = 0
#     builder.request_attribute("delta_2H")
#     return builder.build(attributes=attributes, products=())
#
#
# def do_one_step(formulae, particulator, evaporated_mass_fraction):
#     initial_conc_vap = (
#         formulae.saturation_vapour_pressure.pvs_water(particulator.environment["T"][0])
#         * particulator.environment["RH"][0]
#         / formulae.constants.R_str
#         / particulator.environment["T"][0]
#     )
#     initial_R_vap = (
#         formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
#             molar_mixing_ratio=particulator.environment["molar mixing ratio 2H"][0],
#             density_dry_air=particulator.environment["dry_air_density"][0],
#             conc_vap_total=initial_conc_vap,
#         )
#     )
#     initial_R_liq = (
#         particulator.attributes["moles_2H"][0] / particulator.attributes["moles_1H"][0]
#     )
#
#     dm = -evaporated_mass_fraction * (
#         particulator.attributes["signed water mass"][0]
#         * particulator.attributes["multiplicity"][0]
#     )
#     particulator.attributes["diffusional growth mass change"].data[0] = (
#         dm / particulator.attributes["multiplicity"]
#     )
#     assert np.all(particulator.attributes["diffusional growth mass change"].data < 0)
#
#     particulator.dynamics["IsotopicFractionation"]()
#
#     new_R_vap = (
#         formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
#             molar_mixing_ratio=particulator.environment["molar mixing ratio 2H"].data[
#                 0
#             ],
#             density_dry_air=particulator.environment["dry_air_density"][0],
#             conc_vap_total=initial_conc_vap
#             - dm / formulae.constants.Mv / particulator.environment.mesh.dv,
#         )
#     )
#     new_R_liq = (
#         particulator.attributes["moles_2H"][0] / particulator.attributes["moles_1H"][0]
#     )
#     dR_vap = new_R_vap - initial_R_vap
#     dR_liq = new_R_liq - initial_R_liq
#     return dR_vap / initial_R_vap, dR_liq / initial_R_liq

BASE_INITIAL_ATTRIBUTES = {
    "multiplicity": np.ones(1),
    "dry volume": np.nan,
    "kappa times dry volume": np.nan,
    "signed water mass": np.ones(1) * si.ng,
    **{f"moles_{isotope}": 0 * si.mole for isotope in HEAVY_ISOTOPES},
}


def make_particulator(backend_instance, isotopes_considered, attributes):
    builder = Builder(
        n_sd=1,
        backend=backend_instance,
        environment=Box(dv=np.nan, dt=1 * si.s),
    )
    for iso in isotopes_considered:
        attributes[f"moles_{iso}"] = np.nan
        builder.request_attribute(f"delta_{iso}")
        builder.particulator.environment[f"molar mixing ratio {iso}"] = np.nan
    builder.particulator.environment["RH"] = np.nan
    builder.particulator.environment["T"] = np.nan
    builder.particulator.environment["dry_air_density"] = np.nan

    builder.add_dynamic(Condensation())
    builder.add_dynamic(IsotopicFractionation(isotopes=isotopes_considered))

    return builder.build(attributes)


@pytest.fixture(scope="session")
def formulae():
    return Formulae(
        isotope_relaxation_timescale="ZabaEtAl",
        isotope_diffusivity_ratios="GrahamsLaw",
        isotope_equilibrium_fractionation_factors="VanHook1968",
        isotope_ratio_evolution="GedzelmanAndArnold1994",
        drop_growth="Mason1971",
    )


def zero_conditions(formulae, T, R_vap):
    b_factor = (
        formulae.drop_growth.Fk(T, K=formulae.constants.K0, lv=formulae.constants.l_tri)
        / formulae.constants.rho_w
        * formulae.saturation_vapour_pressure.pvs_water(T)
        / T
        / formulae.constants.Rv
        * formulae.constants.D0
    )
    alpha_2H = formulae.isotope_equilibrium_fractionation_factors.alpha_l_2H(T)
    R_equilibrium = alpha_2H * R_vap / VSMOW_R_2H
    R_range = np.linspace(R_equilibrium, 1.01, 200)
    return R_range, [
        formulae.isotope_ratio_evolution.saturation_for_zero_dR_condition(
            diff_rat_light_to_heavy=1
            / formulae.isotope_diffusivity_ratios.ratio_2H_heavy_to_light(T),
            iso_ratio_x=val_x,
            iso_ratio_r=R_range * VSMOW_R_2H,
            iso_ratio_v=R_vap,
            b=b_factor,
            alpha_w=alpha_2H,
        )
        for val_x in [R_range * VSMOW_R_2H, R_vap]
    ]


class TestIsotopicFractionation:
    @staticmethod
    @pytest.mark.parametrize(
        "dynamics, context",
        (
            pytest.param(
                (Condensation(), IsotopicFractionation(isotopes=("2H",))), nullcontext()
            ),
            pytest.param(
                (IsotopicFractionation(isotopes=("2H",)),),
                pytest.raises(AssertionError, match="dynamics"),
            ),
            pytest.param(
                (IsotopicFractionation(isotopes=("2H",)), Condensation()),
                pytest.raises(AssertionError, match="dynamics"),
            ),
        ),
    )
    def test_ensure_condensation_executed_before(backend_instance, dynamics, context):
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dv=np.nan, dt=1 * si.s)
        )
        for dynamic in dynamics:
            builder.add_dynamic(dynamic)
        builder.particulator.environment[f"molar mixing ratio 2H"] = np.nan

        # act
        with context:
            builder.build(attributes=BASE_INITIAL_ATTRIBUTES.copy())

    @staticmethod
    @pytest.mark.parametrize(
        "isotope",
        [
            *HEAVY_ISOTOPES,
            *[
                pytest.param(
                    iso, marks=pytest.mark.xfail(reason="Light isotope", strict=True)
                )
                for iso in LIGHT_ISOTOPES
            ],
        ],
    )
    def test_fractionation_implemented_for_isotope(backend_instance, isotope):
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dv=np.nan, dt=-1 * si.s)
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=(isotope,)))
        builder.particulator.environment[f"molar mixing ratio {isotope}"] = np.nan
        builder.build(attributes=BASE_INITIAL_ATTRIBUTES.copy())

    @staticmethod
    def test_call_marks_all_isotopes_as_updated(formulae):
        # arrange
        particulator = make_particulator(
            backend_instance=CPU(formulae=formulae),
            isotopes_considered=("2H",),
            attributes=BASE_INITIAL_ATTRIBUTES.copy(),
        )

        for isotope in HEAVY_ISOTOPES:
            # pylint:disable=protected-access
            assert (
                particulator.attributes._ParticleAttributes__attributes[
                    f"moles_{isotope}"
                ].timestamp
                == particulator.attributes._ParticleAttributes__attributes[
                    "multiplicity"
                ].timestamp
            )

        # act
        particulator.dynamics["IsotopicFractionation"]()

        # assert
        for isotope in HEAVY_ISOTOPES:
            # pylint:disable=protected-access
            assert (
                particulator.attributes._ParticleAttributes__attributes[
                    f"moles_{isotope}"
                ].timestamp
                > particulator.attributes._ParticleAttributes__attributes[
                    "multiplicity"
                ].timestamp
            )

    @staticmethod
    def test_no_isotope_fractionation_if_droplet_size_unchanged(
        formulae, backend_class
    ):
        """neither a bug nor a feature :) - just a simplification (?)"""
        # arrange
        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["moles_2H"] = 44
        attributes["signed water mass"] = 1

        particulator = make_particulator(
            backend_instance=backend_class(formulae=formulae),
            attributes=attributes,
            isotopes_considered=(),
        )

        # act
        particulator.attributes["diffusional growth mass change"].data[:] = 0
        particulator.dynamics[
            "IsotopicFractionation"
        ]()  # TODO: call condensation as well!

        # assert
        assert particulator.attributes["moles_2H"][0] == attributes["moles_2H"]

    @staticmethod
    @pytest.mark.parametrize("molecular_R_liq", np.linspace(0.8, 1, 5) * VSMOW_R_2H)
    def test_initial_condition_for_delta_isotopes(
        backend_class,
        formulae,
        molecular_R_liq,
    ):
        # arrange
        const = formulae.constants
        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["moles_2H"] = formulae.trivia.moles_heavy_atom(
            mass_total=attributes["signed water mass"],
            mass_other_heavy_isotopes=sum(
                attributes[f"moles_{isotope}"]
                for isotope in HEAVY_ISOTOPES
                if isotope != "2H"
            ),
            molar_mass_light_molecule=const.M_1H2_16O,
            molar_mass_heavy_molecule=const.M_2H_1H_16O,
            molecular_isotope_ratio=molecular_R_liq,
            atoms_per_heavy_molecule=1,
        )

        particulator = make_particulator(
            backend_instance=backend_class(formulae=formulae),
            attributes=attributes,
            isotopes_considered=("2H",),
        )
        # act

        R_liq = (
            particulator.attributes["moles_2H"][0]
            / particulator.attributes["moles_1H"][0]
        )
        delta_liq = formulae.trivia.isotopic_ratio_2_delta(R_liq, const.VSMOW_R_2H)

        # assert
        np.testing.assert_approx_equal(
            particulator.attributes["moles_2H"][0],
            attributes["moles_2H"],
            significant=50,
        )
        np.testing.assert_approx_equal(
            particulator.attributes["delta_2H"][0],
            delta_liq,
            significant=10,
        )
