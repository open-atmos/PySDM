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
    "dry volume": np.array(1),
    "kappa times dry volume": np.array(np.nan),
    "signed water mass": np.array(np.nan),
    **{f"moles_{isotope}": np.zeros(1) * si.mole for isotope in HEAVY_ISOTOPES},
}


def make_particulator(
    backend_instance,
    isotopes_considered,
    attributes,
    rh=np.nan,
    t=np.nan,
):
    """return basic particulator needed for testing"""
    builder = Builder(
        n_sd=1,
        backend=backend_instance,
        environment=Box(dv=np.ones(1), dt=1 * si.s),
    )
    for iso in isotopes_considered:
        if not attributes.get(f"moles_{iso}"):
            attributes[f"moles_{iso}"] = np.array(0)
        builder.particulator.environment[f"molality {iso} in dry air"] = np.array(0.1)
        builder.request_attribute(f"delta_{iso}")
    builder.particulator.environment["RH"] = np.array(rh)
    builder.particulator.environment["T"] = np.array(t)
    builder.particulator.environment["dry_air_density"] = np.array(1)
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
    def test_fractionation_implemented_for_isotope(backend_class, isotope):
        """test isotopic fractionation implemented
        for heavy water isotopes and raising error for light ones"""

        # arrange
        builder = Builder(
            n_sd=1, backend=backend_class(), environment=Box(dv=np.nan, dt=-1 * si.s)
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=(isotope,)))
        builder.particulator.environment[f"molality {isotope} in dry air"] = np.nan
        builder.build(attributes=BASE_INITIAL_ATTRIBUTES.copy())

    @staticmethod
    @pytest.mark.parametrize("considered_isotopes", (HEAVY_ISOTOPES, ("2H",)))
    def test_call_marks_all_isotopes_as_updated(
        formulae, backend_class, considered_isotopes
    ):
        """test isotopic fractionation dynamic updates moles attribute"""
        if backend_class.__name__ != "Numba":
            pytest.skip("# TODO #1787 - isotopes on GPU")

        # arrange
        particulator = make_particulator(
            backend_instance=backend_class(formulae=formulae),
            isotopes_considered=considered_isotopes,
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
        for isotope in considered_isotopes:
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
    @pytest.mark.parametrize("relative_humidity", (0.1, 0.9, 1.1))
    @pytest.mark.parametrize("temperature", (270 * si.K, 300 * si.K))
    def test_no_isotope_fractionation_if_droplet_size_unchanged(
        formulae, backend_class, relative_humidity, temperature
    ):
        """neither a bug nor a feature :) - just a simplification (?)"""
        if backend_class.__name__ != "Numba":
            pytest.skip("# TODO #1787 - isotopes on GPU")

        # arrange
        attributes = BASE_INITIAL_ATTRIBUTES.copy()
        attributes["moles_2H"] = np.array([44.0])
        attributes["signed water mass"] = np.array([666.0])
        particulator = make_particulator(
            backend_instance=backend_class(formulae=formulae),
            attributes=attributes,
            isotopes_considered=("2H",),
            rh=relative_humidity,
            t=temperature,
        )

        # act
        particulator.attributes["diffusional growth mass change"][:] = 0
        particulator.dynamics["IsotopicFractionation"]()

        # assert
        assert particulator.attributes["moles_2H"].data == attributes["moles_2H"]

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
        if backend_class.__name__ != "Numba":
            pytest.skip("# TODO #1787 - isotopes on GPU")

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
        np.testing.assert_allclose(
            particulator.attributes["moles_2H"][0],
            attributes["moles_2H"][0],
            rtol=1e-12,
            atol=0,
        )
        np.testing.assert_allclose(
            particulator.attributes["delta_2H"][0],
            delta_liq,
            rtol=1e-10,
            atol=0,
        )
