# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Particulator
from PySDM.environments import Box


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_radius(volume, backend_instance):
    # arrange
    particulator = Particulator(
        n_sd=volume.size,
        environment=Box(dt=None, dv=None, backend=backend_instance),
        attributes={"volume": volume, "multiplicity": np.ones_like(volume)},
        requested_attributes=("radius",),
    )

    # act
    radius_actual = particulator.attributes["radius"].to_ndarray()

    # assert
    radius_expected = particulator.formulae.trivia.radius(volume=volume)
    np.testing.assert_allclose(radius_actual, radius_expected)


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_sqrt_radius(volume, backend_instance):
    # arrange
    particulator = Particulator(
        n_sd=volume.size,
        environment=Box(dt=None, dv=None, backend=backend_instance),
        requested_attributes=(
            "radius",
            "square root of radius",
        ),
        attributes={"volume": volume, "multiplicity": np.ones_like(volume)},
    )

    # act
    sqrt_radius_actual = particulator.attributes["square root of radius"].to_ndarray()
    radius_actual = particulator.attributes["radius"].to_ndarray()

    # assert
    sqrt_radius_expected = np.sqrt(radius_actual)
    np.testing.assert_allclose(sqrt_radius_actual, sqrt_radius_expected)


@pytest.mark.parametrize("volume", (np.asarray([44, 666]),))
def test_area(volume, backend_instance):
    # arrange
    particulator = Particulator(
        n_sd=volume.size,
        environment=Box(dv=None, dt=None, backend=backend_instance),
        requested_attributes=("area",),
        attributes={"volume": volume, "multiplicity": np.ones_like(volume)},
    )

    # act
    area_actual = particulator.attributes["area"].to_ndarray()

    # assert
    radius_expected = particulator.formulae.trivia.radius(volume=volume)
    area_expected = particulator.formulae.trivia.area(radius=radius_expected)
    np.testing.assert_allclose(area_actual, area_expected, rtol=1e-6)
