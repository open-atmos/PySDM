# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.environments import Box


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_radius(volume, backend_instance):
    # arrange
    env = Box(dt=None, dv=None)
    builder = Builder(backend=backend_instance, n_sd=volume.size, environment=env)
    builder.request_attribute("radius")
    particulator = builder.build(
        attributes={"volume": volume, "multiplicity": np.ones_like(volume)}
    )

    # act
    radius_actual = particulator.attributes["radius"].to_ndarray()

    # assert
    radius_expected = particulator.formulae.trivia.radius(volume=volume)
    np.testing.assert_allclose(radius_actual, radius_expected)


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_sqrt_radius(volume, backend_instance):
    # arrange
    env = Box(dt=None, dv=None)
    builder = Builder(backend=backend_instance, n_sd=volume.size, environment=env)
    builder.request_attribute("radius")
    builder.request_attribute("square root of radius")
    particulator = builder.build(
        attributes={"volume": volume, "multiplicity": np.ones_like(volume)}
    )

    # act
    radius_actual = particulator.attributes["radius"].to_ndarray()
    sqrt_radius_actual = particulator.attributes["square root of radius"].to_ndarray()

    # assert
    sqrt_radius_expected = np.sqrt(radius_actual)
    np.testing.assert_allclose(sqrt_radius_actual, sqrt_radius_expected)


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_area(volume, backend_instance):
    # arrange
    env = Box(dv=None, dt=None)
    builder = Builder(backend=backend_instance, n_sd=volume.size, environment=env)
    builder.request_attribute("area")
    particulator = builder.build(
        attributes={"volume": volume, "multiplicity": np.ones_like(volume)}
    )

    # act
    area_actual = particulator.attributes["area"].to_ndarray()

    # assert
    radius_expected = particulator.formulae.trivia.radius(volume=volume)
    area_expected = particulator.formulae.trivia.area(radius=radius_expected)
    np.testing.assert_allclose(area_actual, area_expected)
