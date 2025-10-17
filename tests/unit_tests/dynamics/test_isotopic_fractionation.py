"""
unit test for the IsotopicFractionation dynamic
"""

from contextlib import nullcontext

import numpy as np
from matplotlib import pyplot
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.physics.constants_defaults import VSMOW_R_2H

BASE_INITIAL_ATTRIBUTES = {
    "multiplicity": np.ones(1),
    "signed water mass": 1 * si.kg,
    "dry volume": np.nan,
    "kappa times dry volume": np.nan,
    **{f"moles_{isotope}": 0 * si.mole for isotope in HEAVY_ISOTOPES},
}


def prepare_builder_for_build(*, RH, backend_class, isotopes_considered=HEAVY_ISOTOPES):
    formulae = Formulae(
        isotope_relaxation_timescale="GedzelmanAndArnold1994",
        isotope_diffusivity_ratios="GrahamsLaw",
        isotope_equilibrium_fractionation_factors="VanHook1968",
    )
    settings = {
        "mass_dry_air": 1 * si.ng,
        "cell_volume": 1 * si.m**3,
        "dt": -1 * si.s,
        "temperature": formulae.trivia.C2K(10) * si.K,
    }

    builder = Builder(
        n_sd=1,
        backend=backend_class(
            formulae=formulae,
        ),
        environment=Box(dv=settings["cell_volume"], dt=settings["dt"]),
    )
    builder.add_dynamic(Condensation())
    builder.add_dynamic(IsotopicFractionation(isotopes=isotopes_considered))
    builder.request_attribute("delta_2H")
    builder.particulator.environment["T"] = settings["temperature"]
    builder.particulator.environment["RH"] = RH
    builder.particulator.environment["dry_air_density"] = (
        settings["mass_dry_air"] / settings["cell_volume"]
    )
    for isotope in HEAVY_ISOTOPES:
        builder.particulator.environment[f"molar mixing ratio {isotope}"] = 0
    return builder, formulae, settings


def d_delta_rain(
    *, RH, molecular_R_rain, backend_class, isotopes_considered=HEAVY_ISOTOPES
):
    builder, formulae, settings = prepare_builder_for_build(
        RH=RH, backend_class=backend_class, isotopes_considered=isotopes_considered
    )
    const = formulae.constants
    d_mixing_ratio_env = -1

    # initial_R_vap = formulae.trivia.isotopic_delta_2_ratio(
    #     -200 * PER_MILLE, const.VSMOW_R_2H
    # )

    attributes = BASE_INITIAL_ATTRIBUTES.copy()
    moles_total = attributes["signed water mass"] / const.Mv
    heavy_moles_sum_without_2H = sum(
        attributes[f"moles_{isotope}"] for isotope in HEAVY_ISOTOPES
    )
    attributes["moles_2H"] = (
        moles_total - heavy_moles_sum_without_2H
    ) * formulae.trivia.mixing_ratio_to_specific_content(molecular_R_rain)

    builder.request_attribute("delta_2H")
    particulator = builder.build(attributes=attributes, products=())

    for isotope in HEAVY_ISOTOPES:
        if isotope != "2H":
            np.testing.assert_equal(particulator.attributes[f"moles_{isotope}"][0], 0)

    R_rain = (
        particulator.attributes["moles_2H"][0] / particulator.attributes["moles_1H"][0]
    )
    delta_rain = formulae.trivia.isotopic_ratio_2_delta(R_rain, const.VSMOW_R_2H)

    # act
    droplet_dm = (
        -d_mixing_ratio_env * settings["mass_dry_air"] / attributes["multiplicity"]
    )
    particulator.attributes["diffusional growth mass change"].data[:] = droplet_dm
    particulator.dynamics["IsotopicFractionation"]()

    # n_vap_total = 0
    # new_R_vap = (
    #     formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
    #         molar_mixing_ratio=particulator.environment["molar mixing ratio 2H"].data[
    #             0
    #         ],
    #         n_vap_total=n_vap_total,
    #         mass_dry_air=particulator.environment["dry_air_density"].data[0]
    #         * cell_volume,
    #     )
    # )

    return particulator.attributes["delta_2H"][0] - delta_rain


class TestIsotopicFractionation:
    @staticmethod
    @pytest.mark.parametrize(
        "dynamics, context",
        (
            pytest.param(
                (Condensation(), IsotopicFractionation(isotopes=("1H",))), nullcontext()
            ),
            pytest.param(
                (IsotopicFractionation(isotopes=("1H",)),),
                pytest.raises(AssertionError, match="dynamics"),
            ),
            pytest.param(
                (IsotopicFractionation(isotopes=("1H",)), Condensation()),
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

        # act
        with context:
            builder.build(attributes=BASE_INITIAL_ATTRIBUTES.copy())

    @staticmethod
    def test_call_marks_all_isotopes_as_updated(backend_class):
        # arrange
        RH = np.nan
        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        builder, _, _ = prepare_builder_for_build(
            RH=RH,
            backend_class=backend_class,
        )
        particulator = builder.build(attributes=attributes, products=())

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
    def test_no_isotope_fractionation_if_droplet_size_unchanged(backend_class):
        """neither a bug nor a feature :) - just a simplification (?)"""
        # arrange
        builder, _, _ = prepare_builder_for_build(
            RH=np.nan, backend_class=backend_class, isotopes_considered=()
        )
        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["moles_2H"] = 44
        particulator = builder.build(attributes=attributes, products=())

        # act
        particulator.attributes["diffusional growth mass change"].data[:] = 0
        particulator.dynamics[
            "IsotopicFractionation"
        ]()  # TODO: call condensation as well!

        # assert
        assert particulator.attributes["moles_2H"][0] == attributes["moles_2H"]

    @staticmethod
    @pytest.mark.parametrize("molecular_R_rain", np.linspace(0.8, 1, 5) * VSMOW_R_2H)
    def test_initial_condition_for_delta_isotopes(
        backend_class,
        molecular_R_rain,
    ):
        # arrange
        builder, formulae, _ = prepare_builder_for_build(
            RH=1, backend_class=backend_class, isotopes_considered=("2H",)
        )
        const = formulae.constants

        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        heavy_moles_sum_without_2H = sum(
            attributes[f"moles_{isotope}"] for isotope in HEAVY_ISOTOPES
        )
        print(heavy_moles_sum_without_2H)
        attributes["moles_2H"] = (
            attributes["signed water mass"] / const.Mv - heavy_moles_sum_without_2H
        ) * formulae.trivia.mixing_ratio_to_specific_content(molecular_R_rain)

        # act
        builder.request_attribute("delta_2H")
        particulator = builder.build(attributes=attributes, products=())
        R_rain = (
            particulator.attributes["moles_2H"][0]
            / particulator.attributes["moles_1H"][0]
        )
        delta_rain = formulae.trivia.isotopic_ratio_2_delta(R_rain, const.VSMOW_R_2H)

        # assert
        np.testing.assert_approx_equal(
            particulator.attributes["moles_2H"][0],
            attributes["moles_2H"],
            significant=50,
        )
        np.testing.assert_approx_equal(
            particulator.attributes["delta_2H"][0],
            delta_rain,
            significant=10,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "RH, molecular_R_rain, sign_of_dR_vap, sign_of_dR_rain",
        (
            (0.5, 0.86 * VSMOW_R_2H, -1, 1),
            (0.5, 0.9 * VSMOW_R_2H, 1, 1),
            (0.5, 0.98 * VSMOW_R_2H, 1, -1),
        ),
    )
    def test_scenario_from_gedzelman_fig_2_for_single_superdroplet_and_single_isotope(
        RH,
        molecular_R_rain,
        sign_of_dR_vap,
        sign_of_dR_rain,
        backend_class,
    ):
        # arrange
        delta_delta = d_delta_rain(
            molecular_R_rain=molecular_R_rain,
            RH=RH,
            backend_class=backend_class,
            isotopes_considered=("2H",),
        )

        assert np.sign(delta_delta) == sign_of_dR_rain

    @staticmethod
    def test_plot_dR_rain_2H(backend_class):
        number_of_points = 25
        RH = np.linspace(
            0.0, 0.9, number_of_points
        )  # RH=1 is not good, probably dividing by 0?
        molecular_R_rain = np.linspace(0.8, 1, number_of_points) * VSMOW_R_2H

        R_grid, RH_grid = np.meshgrid(molecular_R_rain, RH)
        Z = np.zeros_like(RH_grid)
        for i in range(len(molecular_R_rain)):
            for j in range(len(RH)):
                temp = d_delta_rain(
                    RH=RH[j],
                    molecular_R_rain=molecular_R_rain[i],
                    backend_class=backend_class,
                )
                Z[i, j] = temp
        fig, ax = pyplot.subplots()

        pcm = ax.pcolormesh(
            R_grid,
            RH_grid,
            Z,
        )
        fig.colorbar(pcm, ax=ax, extend="both")

        pyplot.show()
        print(Z)
        assert False

    # TODO
    @staticmethod
    def test_heavy_isotope_changes_match_bolin_number():
        pass
