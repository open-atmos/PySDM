# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.initialisation.sampling.spatial_sampling import Pseudorandom


@pytest.mark.parametrize(
    "seeds",
    (
        pytest.param(
            (44, 33), marks=pytest.mark.xfail(strict=True), id="different seeds"
        ),
        pytest.param((44, 44), id="same seeds"),
    ),
)
def test_pseudorandom_reproducible(seeds, backend_class):
    # arrange
    assert len(seeds) == 2
    backends = [backend_class(Formulae(seed=seed)) for seed in seeds]
    grid = (12, 13)
    n_sd = 14

    # act
    positions = [
        Pseudorandom.sample(backend=backend, grid=grid, n_sd=n_sd, z_part=None)
        for backend in backends
    ]

    # assert
    np.testing.assert_array_equal(positions[0], positions[1])


@pytest.mark.parametrize(
    "z_range",
    (
        pytest.param([0.0, 1.0], id="full range"),
        pytest.param((0.5, 0.75), id="partial range"),
    ),
)
def test_pseudorandom_zrange(z_range, backend_class):
    # arrange
    assert len(z_range) == 2
    backend = backend_class(Formulae())
    grid = (8,)
    n_sd = 100

    # act
    positions = [
        Pseudorandom.sample(backend=backend, grid=grid, n_sd=n_sd, z_part=z_range)
    ]
    comp = np.ones_like(positions)

    # assert
    np.testing.assert_array_less(positions, comp * z_range[1] * grid[0])
    np.testing.assert_array_less(comp * z_range[0] * grid[0], positions)


@pytest.mark.parametrize(
    "z_range, x_range",
    (
        pytest.param((0.0, 1.0), (0.0, 1.0), id="full range"),
        pytest.param((0.5, 0.75), (0.5, 0.75), id="partial range"),
    ),
)
def test_pseudorandom_x_z_range(z_range, x_range, backend_class):
    # arrange
    assert len(z_range) == 2
    assert len(x_range) == 2
    backend = backend_class(Formulae())
    grid = (8, 8)
    n_sd = 100

    # act
    positions = Pseudorandom.sample(
        backend=backend, grid=grid, n_sd=n_sd, z_part=z_range, x_part=x_range
    )

    for droplet in range(n_sd):
        # assert z positions
        np.testing.assert_array_less(positions[0][droplet], z_range[1] * grid[0])
        np.testing.assert_array_less(z_range[0] * grid[0], positions[0][droplet])

        # assert x positions
        np.testing.assert_array_less(positions[1][droplet], x_range[1] * grid[1])
        np.testing.assert_array_less(x_range[0] * grid[1], positions[1][droplet])
