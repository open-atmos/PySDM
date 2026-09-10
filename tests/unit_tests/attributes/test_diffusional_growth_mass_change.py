"""test attribute diffusional growth mass change"""

import numpy as np
import pytest

from PySDM.physics import si
from PySDM.dynamics import Collision
from PySDM.attributes.physics import diffusional_growth_mass_change
from ..dummy_particulator import DummyParticulator


class TestDiffusionalGrowthMassChange:
    @staticmethod
    def test_initialisation(backend_class):
        if backend_class.__name__ != "Numba":
            pytest.skip("only Numba supported - TODO #1438")

        # arrange
        particulator = DummyParticulator(
            backend_class,
            n_sd=1,
            requested_attributes=("diffusional growth mass change",),
            attributes={"multiplicity": np.ones(1), "water mass": np.ones(1)},
        )
        # assert
        for items in (particulator.initialisers, particulator.observers):
            assert len(items) == 1
            assert isinstance(items[0], DiffusionalGrowthMassChange)

    @staticmethod
    @pytest.mark.xfail(
        reason="Not implemented for Collisions",
        raises=AssertionError,
        strict=True,
    )
    def test_if_collision(backend_class):
        _ = DummyParticulator(
            backend_class,
            dynamics=(
                Collision(
                    collision_kernel=np.nan,
                    breakup_efficiency=np.nan,
                    coalescence_efficiency=np.nan,
                    fragmentation_function=np.nan,
                ),
            ),
            attributes={},
            requested_attributes=("diffusional growth mass change",),
        )

    @staticmethod
    @pytest.mark.parametrize("steps", (0, 1, 2))
    def test_methods(backend_class, steps):
        if backend_class.__name__ != "Numba":
            pytest.skip("only Numba supported - TODO #1438")

        # arrange
        n_sd = 1
        mass_delta = np.ones(n_sd) * si.ng

        particulator = DummyParticulator(
            backend_class,
            n_sd=n_sd,
            formulae=None,
            requested_attributes=("diffusional growth mass change",),
            attributes={
                "multiplicity": np.ones(n_sd),
                "water mass": np.ones(n_sd) * si.ug,
            },
        )

        # act
        particulator.run(steps=0)
        for _ in range(steps):
            particulator.attributes["signed water mass"].data[:] += mass_delta
            particulator.run(steps=1)

        # assert
        np.testing.assert_allclose(
            desired=mass_delta if steps != 0 else np.zeros_like(mass_delta),
            actual=particulator.attributes["diffusional growth mass change"].data,
            rtol=1e-8,
        )
