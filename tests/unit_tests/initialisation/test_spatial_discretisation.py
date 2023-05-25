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
        Pseudorandom.sample(backend=backend, grid=grid, n_sd=n_sd)
        for backend in backends
    ]

    # assert
    np.testing.assert_array_equal(positions[0], positions[1])
