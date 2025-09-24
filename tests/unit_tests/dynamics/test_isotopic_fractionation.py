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
    attr: np.asarray([np.nan if attr != "multiplicity" else 0])
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
    def test_ensure_condensation_executed_before(backend_instance, dynamics, context):
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
    def test_call_marks_all_isotopes_as_updated(backend_instance):
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dv=np.nan, dt=1 * si.s)
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
        ambient_initial_isotope_mixing_ratio = 44.0
        particle_initial_isotope_content = 666.0 * si.moles
        cell_volume = 1 * si.m**3

        attributes = DUMMY_ATTRIBUTES.copy()
        attributes["moles_2H"] = particle_initial_isotope_content
        attributes["signed water mass"] = 1 * si.ng
        attributes["multiplicity"] = np.ones(1)

        builder = Builder(
            n_sd=1,
            backend=backend_class(
                formulae=Formulae(isotope_relaxation_timescale="MiyakeEtAl1968"),
            ),
            environment=Box(dv=cell_volume, dt=-1 * si.s),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=("2H",)))
        particulator = builder.build(attributes=attributes, products=())
        particulator.environment["RH"] = 1
        particulator.environment["dry_air_density"] = 1 * si.kg / si.m**3
        particulator.environment["mixing_ratio_2H"] = (
            ambient_initial_isotope_mixing_ratio
        )
        assert (
            particulator.attributes["moles_2H"][0] == particle_initial_isotope_content
        )

        # act
        particulator.attributes["diffusional growth mass change"].data[:] = 0
        particulator.dynamics[
            "IsotopicFractionation"
        ]()  # TODO: call condensation as well!

        # assert
        assert (
            particulator.environment["mixing_ratio_2H"][0]
            == ambient_initial_isotope_mixing_ratio
        )
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
        formulae = Formulae(isotope_relaxation_timescale="MiyakeEtAl1968")
        const = formulae.constants

        multiplicity = np.ones(1)
        m_t = 1 * si.ng
        cell_volume = 1 * si.m**3
        mass_dry_air = 1 * si.kg
        d_mixing_ratio_env = 0.01 * si.g / si.kg
        temperature = formulae.trivia.C2K(10) * si.K
        R_vap = formulae.trivia.isotopic_delta_2_ratio(-200, const.VSMOW_R_2H)

        e = RH * formulae.saturation_vapour_pressure.pvs_water(temperature)
        n_vap_total = e * cell_volume / const.R_str / temperature
        n_vap_heavy = n_vap_total * formulae.trivia.mixing_ratio_to_specific_content(
            R_vap
        )
        mass_2H_vap = n_vap_heavy * const.M_2H

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
        particulator.environment["dry_air_density"] = mass_dry_air / cell_volume
        particulator.environment["mixing_ratio_2H"] = mass_2H_vap / mass_dry_air

        # sanity check for initial condition
        np.testing.assert_approx_equal(
            particulator.attributes["delta_2H"][0], delta_rain, significant=5
        )

        # act
        droplet_dm = -d_mixing_ratio_env * mass_dry_air / multiplicity
        particulator.attributes["diffusional growth mass change"].data[:] = droplet_dm
        particulator.dynamics["IsotopicFractionation"]()

        new_R_vap = particulator.environment["mixing_ratio_2H"][0]
        new_delta_rain = particulator.attributes["delta_2H"][0]

        # assert
        assert np.sign(new_R_vap - R_vap) == sign_of_dR_vap
        assert np.sign(new_delta_rain - delta_rain) == sign_of_dR_rain

    # TODO
    @staticmethod
    def test_heavy_isotope_changes_match_bolin_number():
        pass
