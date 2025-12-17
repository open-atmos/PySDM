"""test attribute diffusional growth mass change"""

import numpy as np
import pytest

from PySDM.attributes import DiffusionalGrowthMassChange
from PySDM.attributes.impl import DummyAttribute
from PySDM.physics import si
from ..dummy_particulator import DummyParticulator
from PySDM.dynamics import Collision


class TestDiffusionalGrowthMassChange:
    @staticmethod
    def test_initialisation(backend_class):
        # arrange
        particulator = DummyParticulator(backend_class)
        particulator.request_attribute("diffusional growth mass change")
        particulator.initialisers = []
        particulator.observers = []

        # act
        particulator.build(
            attributes={"multiplicity": np.ones(1), "water mass": 1 * si.ng}
        )

        # assert
        assert len(particulator.initialisers) == 1
        assert len(particulator.observers) == 1

    @staticmethod
    @pytest.mark.xfail(
        reason="Not implemented for Collisions",
        raises=AssertionError,
        strict=True,
    )
    def test_if_collision(backend_class):
        # arrange
        particulator = DummyParticulator(backend_class)
        particulator.request_attribute("diffusional growth mass change")

        # act
        particulator.add_dynamic(
            Collision(
                collision_kernel=np.nan,
                breakup_efficiency=np.nan,
                coalescence_efficiency=np.nan,
                fragmentation_function=np.nan,
            )
        )

        # assert
        particulator.build(attributes={})

    @staticmethod
    def test_methods(backend_class):
        # arrange
        particulator = DummyParticulator(backend_class)
        particulator.request_attribute("diffusional growth mass change")

        # act
        particulator.build(
            attributes={"multiplicity": np.ones(1), "water mass": np.ones(1) * si.ng}
        )
        sut = DiffusionalGrowthMassChange(particulator)

        # assert
        sut.notify()  # TODO
