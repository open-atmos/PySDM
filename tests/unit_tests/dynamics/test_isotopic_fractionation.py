"""
unit test for the IsotopicFractionation dynamic
"""

from contextlib import nullcontext
from functools import partial

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
    "dry volume": np.nan,
    "kappa times dry volume": np.nan,
    **{f"moles_{isotope}": 0 * si.mole for isotope in HEAVY_ISOTOPES},
}


@pytest.fixture(scope="session")
def formulae():
    return Formulae(
        isotope_relaxation_timescale="GedzelmanAndArnold1994",
        isotope_diffusivity_ratios="GrahamsLaw",
        isotope_equilibrium_fractionation_factors="VanHook1968",
        isotope_ratio_evolution="GedzelmanAndArnold1994",
        drop_growth="Mason1971",
    )


def make_particulator(
    *,
    formulae,
    backend_class,
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
    )
    builder = Builder(
        n_sd=n_sd,
        backend=backend_class(
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


def do_one_step(formulae, particulator, evaporated_mass_fraction):
    initial_conc_vap = (
        formulae.saturation_vapour_pressure.pvs_water(particulator.environment["T"][0])
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
        particulator.attributes["moles_2H"][0] / particulator.attributes["moles_1H"][0]
    )

    dm = -evaporated_mass_fraction * (
        particulator.attributes["signed water mass"][0]
        * particulator.attributes["multiplicity"][0]
    )
    particulator.attributes["diffusional growth mass change"].data[0] = (
        dm / particulator.attributes["multiplicity"]
    )
    assert np.all(particulator.attributes["diffusional growth mass change"].data < 0)

    particulator.dynamics["IsotopicFractionation"]()

    new_R_vap = (
        formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
            molar_mixing_ratio=particulator.environment["molar mixing ratio 2H"].data[
                0
            ],
            density_dry_air=particulator.environment["dry_air_density"][0],
            conc_vap_total=initial_conc_vap
            - dm / formulae.constants.Mv / particulator.environment.mesh.dv,
        )
    )
    new_R_liq = (
        particulator.attributes["moles_2H"][0] / particulator.attributes["moles_1H"][0]
    )
    dR_vap = new_R_vap - initial_R_vap
    dR_liq = new_R_liq - initial_R_liq
    return dR_vap / initial_R_vap, dR_liq / initial_R_liq


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
    def test_call_marks_all_isotopes_as_updated(backend_class, formulae):
        # arrange
        particulator = make_particulator(
            formulae=formulae,
            backend_class=backend_class,
            isotopes_considered=HEAVY_ISOTOPES,
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
            formulae=formulae,
            backend_class=backend_class,
            attributes=attributes,
            isotopes_considered=(),
            dv=1 * si.m**3,
            T=formulae.trivia.C2K(10) * si.K,
            RH=1,
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
            molecular_R_liq=molecular_R_liq,
            mass_total=attributes["signed water mass"],
            mass_other_heavy_isotopes=sum(
                attributes[f"moles_{isotope}"]
                for isotope in HEAVY_ISOTOPES
                if isotope != "2H"
            ),
            water_molar_mass=const.Mv,
            molar_mass_heavy_molecule=const.M_2H_1H_16O,
        )

        particulator = make_particulator(
            formulae=formulae,
            backend_class=backend_class,
            attributes=attributes,
            RH=1,
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
    @pytest.mark.parametrize("multiplicity", (1e0, 1e3, 1e6))
    @pytest.mark.parametrize("volume", (1 * si.m**3, 1000 * si.m**3))
    def test_plot_dR_liq_2H(
        formulae,
        backend_class,
        multiplicity,
        volume,
        n_sd=1,
        liquid_water_content=1 * si.g / si.m**3,
        evaporated_mass_fraction=0.02,
    ):
        const = formulae.constants
        T = formulae.trivia.C2K(10) * si.K
        delta_2H = -200 * PER_MILLE
        initial_R_vap = {
            "2H": formulae.trivia.isotopic_delta_2_ratio(delta_2H, const.VSMOW_R_2H)
        }
        grid = (6, 6)
        molecular_R_liq = np.linspace(0.8, 1, grid[0]) * VSMOW_R_2H
        RH = np.linspace(0.1, 1.0, grid[1])

        total_liquid_water_mass = liquid_water_content * volume

        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["multiplicity"] = multiplicity * np.ones(n_sd)
        attributes["signed water mass"] = total_liquid_water_mass / multiplicity

        rel_diff_vap = np.zeros(grid)
        rel_diff_liq = np.zeros(grid)

        for i in range(grid[0]):
            for j in range(grid[1]):
                rel_diff_vap[i, j], rel_diff_liq[i, j] = do_one_step(
                    formulae=formulae,
                    particulator=make_particulator(
                        RH=RH[i],
                        molecular_R_liq=molecular_R_liq[j],
                        backend_class=backend_class,
                        formulae=formulae,
                        attributes=attributes,
                        initial_R_vap=initial_R_vap,
                        dv=volume,
                        n_sd=n_sd,
                        T=T,
                    ),
                    evaporated_mass_fraction=evaporated_mass_fraction,
                )
        phases = ["liquid", "vapour"]
        R_grid, RH_percent_grid = np.meshgrid(
            molecular_R_liq / VSMOW_R_2H, in_unit(RH, PER_CENT)
        )
        fig, ax = pyplot.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        fig.suptitle(f"$\\xi$={multiplicity:.0e}, V={volume:.0e} [m$^3$]")
        for idx, data in enumerate([rel_diff_liq, rel_diff_vap]):
            pcm = ax[idx].pcolormesh(
                R_grid,
                RH_percent_grid,
                data_percent := in_unit(data, PER_CENT),
                cmap="seismic",
                vmax=(vmax := np.nanmax(np.abs(data_percent))),
                vmin=-vmax,
            )
            fig.colorbar(pcm, ax=ax[idx], extend="both")
            ax[idx].set_title(f"$\\Delta R/R$ [%], R-{phases[idx]}")
            ax[idx].set_xlabel(
                "molecular isotope ratio of rain normalised to atomic VSMOW [1]"
            )

        R_range, zero_cond = zero_conditions(formulae, T, initial_R_vap["2H"])
        for ax_i, y, name in zip(ax, zero_cond, phases):
            ax_i.plot(R_range, in_unit(y, PER_CENT), label=f"{name} line from GA")
            ax_i.legend()
            ax_i.set_ylim(in_unit(RH[0], PER_CENT), in_unit(RH[-1], PER_CENT))

        pyplot.tight_layout()
        show_plot("R_ratios_for_liquid_and_vapour")

    # TODO
    @staticmethod
    def test_heavy_isotope_changes_match_bolin_number():
        pass
