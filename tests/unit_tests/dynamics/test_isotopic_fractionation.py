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


DUMMY_ATTRIBUTES = {
    attr: np.asarray([np.nan if attr != "multiplicity" else 0])
    for attr in (
        "multiplicity",
        "water mass",
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
        attributes["water mass"] = 1 * si.ng
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
