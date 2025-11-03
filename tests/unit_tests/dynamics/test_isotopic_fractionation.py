"""
unit test for the IsotopicFractionation dynamic
"""

from contextlib import nullcontext

import numpy as np
from matplotlib import pyplot
import pytest
from open_atmos_jupyter_utils import show_plot

from PySDM import Builder, Formulae
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import si, in_unit
from PySDM.physics.constants import PER_MILLE, PER_CENT
from PySDM.physics.constants_defaults import VSMOW_R_2H  # TODO!

BASE_INITIAL_ATTRIBUTES = {
    "multiplicity": np.ones(1),
    "signed water mass": np.nan,
    "dry volume": np.nan,
    "kappa times dry volume": np.nan,
    **{f"moles_{isotope}": 0 * si.mole for isotope in HEAVY_ISOTOPES},
}


def set_up_env_and_do_one_step(
    *,
    RH,
    molecular_R_liq,
    backend_class,
    formulae,
    attributes,
    initial_R_vap,
):
    const = formulae.constants
    cell_volume = 1 * si.cm**3
    T = formulae.trivia.C2K(10) * si.K
    builder = Builder(
        n_sd=1,
        backend=backend_class(
            formulae=formulae,
        ),
        environment=Box(dv=cell_volume, dt=-1 * si.s),
    )
    builder.add_dynamic(Condensation())
    builder.add_dynamic(IsotopicFractionation(isotopes=("2H",)))
    builder.particulator.environment["T"] = T
    builder.particulator.environment["RH"] = RH
    builder.particulator.environment["dry_air_density"] = (
        formulae.constants.p_STP / formulae.constants.Rd / T
    )

    attributes["moles_2H"] = formulae.trivia.moles_heavy_isotope(
        molecular_R_liq=molecular_R_liq,
        mass_total=attributes["signed water mass"],
        mass_other_heavy_isotopes=0,
        water_molar_mass=formulae.constants.Mv,
        molar_mass_heavy_molecule=formulae.constants.M_2H_1H_16O,
    )
    initial_conc_vap = (
        formulae.saturation_vapour_pressure.pvs_water(T) * RH / const.R_str / T
    )
    if initial_R_vap is None:
        initial_R_vap = {}
    for isotope in HEAVY_ISOTOPES:
        initial_R_vap.setdefault(isotope, 0)
        builder.particulator.environment[f"molar mixing ratio {isotope}"] = (
            formulae.trivia.R_vap_to_molar_mixing_ratio_assuming_single_heavy_isotope(
                R_vap=initial_R_vap[isotope],
                density_dry_air=builder.particulator.environment["dry_air_density"][0],
                conc_vap_total=initial_conc_vap,
            )
        )
    builder.request_attribute("delta_2H")
    particulator = builder.build(attributes=attributes, products=())

    initial_R_liq = (
        particulator.attributes["moles_2H"][0] / particulator.attributes["moles_1H"][0]
    )

    dm = -0.1 * particulator.attributes["signed water mass"][0]
    particulator.attributes["diffusional growth mass change"].data[:] = dm
    assert np.all(particulator.attributes["diffusional growth mass change"].data < 0)

    # act
    previous_moles = (
        particulator.attributes["moles_1H"][0],
        particulator.attributes["moles_2H"][0],
    )
    particulator.dynamics["IsotopicFractionation"]()
    current_moles = (
        particulator.attributes["moles_1H"][0],
        particulator.attributes["moles_2H"][0],
    )
    print(
        f"1H: {previous_moles[0] - current_moles[0]}, 2H: {previous_moles[1]-current_moles[1]}"
    )
    print(f"Bolin number: {particulator.attributes['Bolin number for 2H'][0]}")

    new_R_vap = (
        formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
            molar_mixing_ratio=particulator.environment["molar mixing ratio 2H"].data[
                0
            ],
            density_dry_air=particulator.environment["dry_air_density"][0],
            conc_vap_total=initial_conc_vap - dm / const.Mv / cell_volume,
        )
    )
    dR_vap = new_R_vap - initial_R_vap["2H"]
    dR_liq = (
        particulator.attributes["moles_2H"][0] / particulator.attributes["moles_1H"][0]
        - initial_R_liq
    )
    return dR_vap / initial_R_vap["2H"], dR_liq / initial_R_liq


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
        formulae = Formulae(
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="GrahamsLaw",
            isotope_equilibrium_fractionation_factors="VanHook1968",
        )
        builder = prepare_builder_for_build(
            RH=RH,
            backend_class=backend_class,
            cell_volume=np.nan,
            formulae=formulae,
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
        formulae = Formulae(
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="GrahamsLaw",
            isotope_equilibrium_fractionation_factors="VanHook1968",
        )
        cell_volume = 1
        T = formulae.trivia.C2K(10) * si.K
        builder = Builder(
            n_sd=1,
            backend=backend_class(
                formulae=formulae,
            ),
            environment=Box(dv=cell_volume, dt=-1 * si.s),
        )

        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation())
        builder.particulator.environment["T"] = T
        builder.particulator.environment["RH"] = 1
        builder.particulator.environment["dry_air_density"] = 1
        builder.request_attribute("delta_2H")
        for isotope in HEAVY_ISOTOPES:
            builder.particulator.environment[f"molar mixing ratio {isotope}"] = 0

        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["moles_2H"] = 44
        attributes["signed water mass"] = 1

        particulator = builder.build(attributes=attributes, products=())

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
        molecular_R_liq,
    ):
        # arrange
        formulae = Formulae(
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="GrahamsLaw",
            isotope_equilibrium_fractionation_factors="VanHook1968",
        )
        builder = prepare_builder_for_build(
            RH=1,
            backend_class=backend_class,
            isotopes_considered=("2H",),
            cell_volume=np.nan,
            formulae=formulae,
        )
        const = formulae.constants

        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        heavy_moles_sum_without_2H = sum(
            attributes[f"moles_{isotope}"]
            for isotope in HEAVY_ISOTOPES
            if isotope != "2H"
        )

        attributes["moles_2H"] = (
            attributes["signed water mass"] / const.Mv - heavy_moles_sum_without_2H
        ) * formulae.trivia.mixing_ratio_to_specific_content(molecular_R_liq)

        # act
        builder.request_attribute("delta_2H")
        particulator = builder.build(attributes=attributes, products=())
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

    # @staticmethod
    # @pytest.mark.parametrize(
    #     "RH, molecular_R_liq, sign_of_dR_vap, sign_of_dRliq",
    #     (
    #         (0.5, 0.86 * VSMOW_R_2H, -1, 1),
    #         (0.5, 0.9 * VSMOW_R_2H, 1, 1),
    #         (0.5, 0.98 * VSMOW_R_2H, 1, -1),
    #     ),
    # )
    # def test_scenario_from_gedzelman_fig_2_for_single_superdroplet_and_single_isotope(  # TODO
    #     RH,
    #     molecular_R_liq,
    #     sign_of_dR_vap,
    #     sign_of_dR_liq,
    #     backend_class,
    # ):
    #     # arrange
    #     delta_delta = d_delta_liq(
    #         molecular_R_liq=molecular_R_liq,
    #         RH=RH,
    #         backend_class=backend_class,
    #         isotopes_considered=("2H",),
    #     )
    #
    #     assert np.sign(delta_delta) == sign_of_dR_liq

    @staticmethod
    def test_plot_dR_liq_2H(backend_class):
        formulae = Formulae(
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="GrahamsLaw",
            isotope_equilibrium_fractionation_factors="VanHook1968",
        )
        const = formulae.constants
        T = formulae.trivia.C2K(10) * si.K
        initial_R_vap = {
            "2H": formulae.trivia.isotopic_delta_2_ratio(
                -200 * PER_MILLE, const.VSMOW_R_2H
            )
        }
        number_of_points = 8
        molecular_R_liq = np.linspace(0.2, 1, number_of_points) * VSMOW_R_2H
        RH = np.linspace(0, 1, number_of_points)

        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["signed water mass"] = const.rho_w * formulae.trivia.volume(
            radius=0.1 * si.mm
        )

        R_grid, RH_grid = np.meshgrid(
            molecular_R_liq / formulae.constants.VSMOW_R_2H, RH
        )
        rel_diff_vap = np.zeros_like(RH_grid)
        rel_diff_liq = np.zeros_like(RH_grid)

        for i in range(number_of_points):
            for j in range(number_of_points):
                rel_diff_vap[i, j], rel_diff_liq[i, j] = set_up_env_and_do_one_step(
                    RH=RH[i],
                    molecular_R_liq=molecular_R_liq[j],
                    backend_class=backend_class,
                    formulae=formulae,
                    attributes=attributes,
                    initial_R_vap=initial_R_vap,
                )
        R_equilibrium = (
            formulae.isotope_equilibrium_fractionation_factors.alpha_l_2H(T)
            * initial_R_vap["2H"]
            / VSMOW_R_2H
        )
        labels = [
            "$\\Delta R_\\text{liq} / R_\\text{liq}$ [%]",
            "$\\Delta R_\\text{vap} / R_\\text{vap}$ [%]",
        ]
        fig, ax = pyplot.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for i, data in enumerate([rel_diff_liq, rel_diff_vap]):
            pcm = ax[i].pcolormesh(
                R_grid,
                in_unit(RH_grid, PER_CENT),
                data_percent := in_unit(data, PER_CENT),
                cmap="seismic",
                vmin=-np.nanmax(np.abs(data_percent)),
                vmax=np.nanmax(np.abs(data_percent)),
            )
            fig.colorbar(pcm, ax=ax[i], extend="both")

            ax[i].set_title(labels[i])
            ax[i].set_xlabel(
                "molecular isotope ratio of rain normalised to atomic VSMOW [1]"
            )
            ax_top = ax[i].secondary_xaxis("top")

            ax_top.set_xticks([R_equilibrium])
            ax_top.set_xticklabels(["eq"])

            ax_top.tick_params(axis="x", direction="out", length=6)

            pyplot.xticks(molecular_R_liq / formulae.constants.VSMOW_R_2H)
            pyplot.yticks(in_unit(RH, PER_CENT))
        pyplot.tight_layout()
        show_plot("R_ratios_for_liquid_and_vapour")

    # TODO
    @staticmethod
    def test_heavy_isotope_changes_match_bolin_number():
        pass
