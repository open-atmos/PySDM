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

# TODO
R_SMOW = Formulae().constants.VSMOW_R_2H


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
        "RH, R_rain, sign_of_dR_vap, sign_of_dR_rain",
        (
            (0.5, 0.86 * R_SMOW, -1, 1),
            (0.5, 0.9 * R_SMOW, 1, 1),
            (0.5, 0.98 * R_SMOW, 1, -1),
        ),
    )
    def test_scenario_from_gedzelman_fig_2(
        RH, R_rain, sign_of_dR_vap, sign_of_dR_rain, backend_class
    ):
        # arrange
        formulae = Formulae(isotope_relaxation_timescale="MiyakeEtAl1968")
        const = formulae.constants

        n_2H_liq = 666.0 * si.moles
        n_liq = n_2H_liq / R_rain
        cell_volume = 1 * si.m**3
        mass_dry_air = 1 * si.kg
        temperature = formulae.trivia.C2K(10) * si.K
        R_vap = formulae.trivia.isotopic_delta_2_ratio(-200 * PER_MILLE, R_SMOW)
        e = RH * formulae.saturation_vapour_pressure.pvs_water(temperature)
        n_vap = (e + const.p_STP) * cell_volume / const.R_str / temperature
        mass_2H_vap = R_vap * n_vap * const.M_2H_1H_16O
        r_v = mass_2H_vap / mass_dry_air

        attributes = DUMMY_ATTRIBUTES.copy()
        attributes["moles_2H"] = n_2H_liq
        attributes["signed water mass"] = 1 * si.ng
        attributes["multiplicity"] = np.ones(1)

        builder = Builder(
            n_sd=1,
            backend=backend_class(formulae=formulae),
            environment=Box(dv=cell_volume, dt=-1 * si.s),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=("2H",)))
        particulator = builder.build(attributes=attributes, products=())
        particulator.environment["RH"] = RH
        particulator.environment["dry_air_density"] = mass_dry_air / cell_volume
        particulator.environment["mixing_ratio_2H"] = r_v

        # act
        dn_liq_dt = -1
        dm_dt = dn_liq_dt * const.M_1H2_16O
        particulator.attributes["diffusional growth mass change"].data[:] = dm_dt
        particulator.dynamics["IsotopicFractionation"]()

        new_n_liq = dn_liq_dt + n_liq
        new_r_v = particulator.environment["mixing_ratio_2H"][0]
        new_n_2H_liq = particulator.attributes["moles_2H"][0]

        new_R_vap = new_r_v * mass_dry_air / n_vap / const.M_2H_1H_16O  # n_vap??
        new_R_liq = new_n_2H_liq / new_n_liq

        dR_vap = new_R_vap - R_vap
        dR_rain = new_R_liq - R_rain

        # assert
        assert np.sign(dR_vap) == sign_of_dR_vap
        assert np.sign(dR_rain) == sign_of_dR_rain

    # TODO
    @staticmethod
    def test_heavy_isotope_changes_match_bolin_number():
        pass
