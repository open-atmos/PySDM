"""
unit tests for the IsotopicFractionation dynamic
"""

# pylint: disable=redefined-outer-name
from contextlib import nullcontext

import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES, LIGHT_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import si

BASE_INITIAL_ATTRIBUTES = {
    "multiplicity": np.ones(1),
    "dry volume": np.array(np.nan),
    "kappa times dry volume": np.array(np.nan),
    "signed water mass": np.array(np.nan),
    **{f"moles_{isotope}": np.zeros(1) * si.mole for isotope in HEAVY_ISOTOPES},
}


def make_particulator(backend_instance, isotopes_considered, attributes):
    """return basic particulator needed for testing"""
    builder = Builder(
        n_sd=1,
        backend=backend_instance,
        environment=Box(dv=np.nan, dt=1 * si.s),
    )
    for iso in isotopes_considered:
        attributes[f"moles_{iso}"] = np.array(np.nan)
        builder.request_attribute(f"delta_{iso}")
        builder.particulator.environment[f"molality {iso} in dry air"] = np.array(
            np.nan
        )
    builder.particulator.environment["RH"] = np.array(np.nan)
    builder.particulator.environment["T"] = np.array(np.nan)
    builder.particulator.environment["dry_air_density"] = np.array(np.nan)

    builder.add_dynamic(Condensation())
    builder.add_dynamic(IsotopicFractionation(isotopes=isotopes_considered))

    return builder.build(attributes)


@pytest.fixture(scope="session")
def formulae():
    """common formulae"""
    return Formulae(
        isotope_relaxation_timescale="ZabaEtAl",
        isotope_diffusivity_ratios="GrahamsLaw",
        isotope_equilibrium_fractionation_factors="VanHook1968",
        isotope_ratio_evolution="GedzelmanAndArnold1994",
        drop_growth="Mason1971",
    )


class TestIsotopicFractionation:
    """test the IsotopicFractionation dynamic"""

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
        """
        test that run fails when isotopic fractionation
        is executed before or without condensation"""
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dv=np.nan, dt=1 * si.s)
        )
        for dynamic in dynamics:
            builder.add_dynamic(dynamic)
        builder.particulator.environment["molality 2H in dry air"] = np.nan

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
        """test isotopic fractionation implemented
        for heavy water isotopes and raising error for light ones"""
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dv=np.nan, dt=-1 * si.s)
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=(isotope,)))
        builder.particulator.environment[f"molality {isotope} in dry air"] = np.nan
        builder.build(attributes=BASE_INITIAL_ATTRIBUTES.copy())

    @staticmethod
    def test_call_marks_all_isotopes_as_updated(formulae, backend_class):
        """test isotopic fractionation dynamic updates moles attribute"""
        # arrange
        particulator = make_particulator(
            backend_instance=backend_class(formulae=formulae),
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
        attributes["moles_2H"] = 44.0 * np.ones(1)

        particulator = make_particulator(
            backend_instance=backend_class(formulae=formulae),
            attributes=attributes,
            isotopes_considered=(),
        )

        # act
        particulator.attributes["diffusional growth mass change"].data[:] = 0
        particulator.dynamics["IsotopicFractionation"]()

        # assert
        assert particulator.attributes["moles_2H"][0] == attributes["moles_2H"]

    @staticmethod
    @pytest.mark.parametrize(
        "molecular_R_liq",
        np.linspace(0.8, 1, 5) * Formulae().constants.VSMOW_R_2H,
    )
    def test_initial_condition_for_delta_isotopes(
        backend_class,
        formulae,
        molecular_R_liq,
    ):
        """test initial condition for delta_isotopes is calculated properly"""
        # arrange
        const = formulae.constants
        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["signed water mass"] = np.ones(1) * si.ng
        attributes["moles_2H"] = formulae.trivia.moles_heavy_atom(
            mass_total=attributes["signed water mass"],
            mass_other_heavy_isotopes=sum(
                attributes[f"moles_{isotope}"]
                for isotope in HEAVY_ISOTOPES
                if isotope != "2H"
            ),
            molar_mass_light_molecule=const.M_1H2_16O,
            molar_mass_heavy_molecule=const.M_2H_1H_16O,
            molecular_isotopic_ratio=molecular_R_liq,
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
