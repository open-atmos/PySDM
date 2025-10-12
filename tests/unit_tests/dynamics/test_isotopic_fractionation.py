"""
unit test for the IsotopicFractionation dynamic
"""

from contextlib import nullcontext

import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.physics.constants_defaults import PER_MILLE

DUMMY_ATTRIBUTES = {
    attr: np.asarray([0 if attr == "multiplicity" else np.nan])
    for attr in (
        "multiplicity",
        "signed water mass",
        "dry volume",
        "kappa times dry volume",
        *[f"moles_{isotope}" for isotope in HEAVY_ISOTOPES],
    )
}


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
    def test_ensure_condensation_executed_before(
        backend_instance, dynamics, context
    ):  # Q: Why this is working for all dynamics and contexts for any isotopes besides 1H
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dv=np.nan, dt=1 * si.s)
        )
        for dynamic in dynamics:
            builder.add_dynamic(dynamic)

        # act
        with context:
            builder.build(attributes=DUMMY_ATTRIBUTES)

    @staticmethod
    def test_call_marks_all_isotopes_as_updated(backend_class):
        # arrange
        formulae = Formulae(
            isotope_diffusivity_ratios="GrahamsLaw",
            isotope_equilibrium_fractionation_factors="VanHook1968",
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
        )
        builder = Builder(
            n_sd=1,
            backend=backend_class(formulae=formulae),
            environment=Box(dv=np.nan, dt=1 * si.s),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation())
        particulator = builder.build(attributes=DUMMY_ATTRIBUTES, products=())
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
        particulator.environment["T"] = np.nan
        particulator.environment["RH"] = np.nan
        particulator.environment["dry_air_density"] = np.nan

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

        # ambient_initial_isotope_mixing_ratio = 44.0
        # ambient initial isotope mixing ratio
        #   -> R_vap
        #   -> delta
        delta_2H = 44 * PER_MILLE
        particle_initial_isotope_content = 666.0 * si.moles
        cell_volume = 1 * si.m**3

        attributes = DUMMY_ATTRIBUTES.copy()
        attributes["moles_2H"] = particle_initial_isotope_content
        attributes["signed water mass"] = 1 * si.ng
        attributes["multiplicity"] = np.ones(1)

        formulae = Formulae(
            isotope_diffusivity_ratios="GrahamsLaw",
            isotope_equilibrium_fractionation_factors="VanHook1968",
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
        )
        builder = Builder(
            n_sd=1,
            backend=backend_class(
                formulae=formulae,
            ),
            environment=Box(dv=cell_volume, dt=-1 * si.s),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=("2H",)))
        particulator = builder.build(attributes=attributes, products=())
        particulator.environment["T"] = formulae.trivia.C2K(10) * si.K
        particulator.environment["RH"] = 1
        particulator.environment["dry_air_density"] = 1 * si.kg / si.m**3
        particulator.environment["delta_2H"] = delta_2H

        assert (
            particulator.attributes["moles_2H"][0] == particle_initial_isotope_content
        )

        # act
        particulator.attributes["diffusional growth mass change"].data[:] = 0
        particulator.dynamics[
            "IsotopicFractionation"
        ]()  # TODO: call condensation as well!

        # assert
        assert particulator.environment["delta_2H"][0] == delta_2H
        assert (
            particulator.attributes["moles_2H"][0] == particle_initial_isotope_content
        )

    @staticmethod
    @pytest.mark.parametrize(
        "RH, delta_rain, sign_of_dR_vap, sign_of_dR_rain",
        (
            (0.5, 0.86 - 1, -1, 1),
            (0.5, 0.9 - 1, 1, 1),
            (0.5, 0.98 - 1, 1, -1),
        ),
    )
    def test_scenario_from_gedzelman_fig_2_for_single_superdroplet_and_single_isotope(
        RH,
        delta_rain,
        sign_of_dR_vap,
        sign_of_dR_rain,
        backend_class,
        # TODO: multiplicity, dt, ...
    ):
        # arrange
        formulae = Formulae(
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="GrahamsLaw",  # TODO which one
            isotope_equilibrium_fractionation_factors="VanHook1968",  # TODO which one
            saturation_vapour_pressure="FlatauWalkoCotton",  # TODO which one
        )
        const = formulae.constants
        multiplicity = np.ones(1)
        m_t = 1 * si.ng
        cell_volume = 1 * si.m**3
        mass_dry_air = 1 * si.kg
        d_mixing_ratio_env = 1 * si.ng / si.kg
        temperature = formulae.trivia.C2K(10) * si.K
        R_vap = formulae.trivia.isotopic_delta_2_ratio(
            -200 * PER_MILLE, const.VSMOW_R_2H
        )

        n_vap_total = formulae.trivia.n_vap_total(
            RH=RH,
            temperature=temperature,
            pvs_water=formulae.saturation_vapour_pressure.pvs_water(temperature),
            cell_volume=cell_volume,
        )
        mass_2H_vap = n_vap_total * (R_vap / (1 + R_vap)) * const.M_2H_1H_16O

        mixing_ratio_2H = (
            formulae.trivia.R_vap_to_mixing_ratio_assuming_single_heavy_isotope(
                R_vap=R_vap,
                n_vap_total=n_vap_total,
                heavy_molar_mass=mass_2H_vap,
                mass_dry_air=mass_dry_air,
            )
        )
        attributes = DUMMY_ATTRIBUTES.copy()
        attributes["moles_2H"] = formulae.trivia.moles_heavy_atom(
            delta=delta_rain,
            mass_total=m_t,
            molar_mass_heavy_molecule=const.M_2H_1H_16O,
            R_STD=const.VSMOW_R_2H,
            light_atoms_per_light_molecule=2,
        )
        for isotope in HEAVY_ISOTOPES:
            if isotope != "2H":
                attributes[f"moles_{isotope}"] = 0
        attributes["signed water mass"] = m_t
        attributes["multiplicity"] = multiplicity

        builder = Builder(
            n_sd=1,
            backend=backend_class(formulae=formulae),
            environment=Box(dv=cell_volume, dt=-1 * si.s),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=("2H",)))
        builder.request_attribute("delta_2H")
        particulator = builder.build(attributes=attributes, products=())
        particulator.environment["RH"] = RH
        particulator.environment["T"] = temperature
        particulator.environment["dry_air_density"] = mass_dry_air / cell_volume
        particulator.environment["mixing ratio 2H"] = mixing_ratio_2H
        for isotope in HEAVY_ISOTOPES:
            if isotope != "2H":
                particulator.environment[f"mixing ratio {isotope}"] = 0

        # sanity check for initial condition

        np.testing.assert_approx_equal(
            particulator.attributes["delta_2H"][0],
            delta_rain,
            significant=5,
        )

        # act
        droplet_dm = -d_mixing_ratio_env * mass_dry_air / multiplicity
        particulator.attributes["diffusional growth mass change"].data[:] = droplet_dm
        particulator.dynamics["IsotopicFractionation"]()

        n_vap_total = formulae.trivia.n_vap_total(
            RH=particulator.environment["RH"].data,
            temperature=temperature,
            pvs_water=formulae.saturation_vapour_pressure.pvs_water(temperature),
            cell_volume=cell_volume,
        )
        heavy_molar_mass = const.M_2H_1H_16O  # TODO check!!!!!
        new_R_vap = formulae.trivia.mixing_ratio_to_R_vap(
            mixing_ratio=particulator.environment["mixing ratio 2H"].data[0],
            n_vap_total=n_vap_total,
            heavy_molar_mass=heavy_molar_mass,
            mass_dry_air=particulator.environment["dry_air_density"].data[0]
            * cell_volume,
        )
        new_delta_rain = particulator.attributes["delta_2H"][0]

        # assert
        assert np.sign(new_R_vap - R_vap) == sign_of_dR_vap
        assert np.sign(new_delta_rain - delta_rain) == sign_of_dR_rain

    # TODO
    @staticmethod
    def test_heavy_isotope_changes_match_bolin_number():
        pass
