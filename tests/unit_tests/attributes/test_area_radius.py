# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_radius(volume):
    # arrange
    builder = Builder(backend=CPU(), n_sd=volume.size)
    builder.set_environment(Box(dt=None, dv=None))
    builder.request_attribute("radius")
    particulator = builder.build(
        attributes={"volume": [volume], "n": np.ones_like(volume)}
    )

    # act
    radius_actual = particulator.attributes["radius"].to_ndarray()

    # assert
    radius_expected = particulator.formulae.trivia.radius(volume=volume)
    np.testing.assert_allclose(radius_actual, radius_expected)


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_area(volume):
    # arrange
    builder = Builder(backend=CPU(), n_sd=volume.size)
    builder.set_environment(Box(dt=None, dv=None))
    builder.request_attribute("area")
    particulator = builder.build(
        attributes={"volume": [volume], "n": np.ones_like(volume)}
    )

    # act
    area_actual = particulator.attributes["area"].to_ndarray()

    # assert
    radius_expected = particulator.formulae.trivia.radius(volume=volume)
    area_expected = particulator.formulae.trivia.area(radius=radius_expected)
    np.testing.assert_allclose(area_actual, area_expected)
