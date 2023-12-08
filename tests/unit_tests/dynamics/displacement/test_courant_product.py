# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import Displacement
from PySDM.environments import Kinematic2D
from PySDM.products import MaxCourantNumber

GRID = (3, 4)


@pytest.mark.parametrize(
    "courant_field",
    (
        (np.zeros((GRID[0] + 1, GRID[1])), np.zeros((GRID[0], GRID[1] + 1))),
        (np.ones((GRID[0] + 1, GRID[1])), np.zeros((GRID[0], GRID[1] + 1))),
        (np.zeros((GRID[0] + 1, GRID[1])), np.ones((GRID[0], GRID[1] + 1))),
        (-0.3 * np.ones((GRID[0] + 1, GRID[1])), 0.2 * np.ones((GRID[0], GRID[1] + 1))),
    ),
)
def test_courant_product(courant_field):
    # arrange
    n_sd = 1
    env = Kinematic2D(dt=1, grid=GRID, size=(100, 100), rhod_of=lambda x: x * 0 + 1)
    builder = Builder(n_sd=n_sd, backend=CPU(), environment=env)
    builder.add_dynamic(Displacement())
    particulator = builder.build(
        attributes={
            "multiplicity": np.ones(n_sd),
            "volume": np.ones(n_sd),
            "cell id": np.zeros(n_sd, dtype=int),
        },
        products=(MaxCourantNumber(),),
    )
    sut = particulator.products["max courant number"]

    # act
    particulator.dynamics["Displacement"].upload_courant_field(courant_field)
    max_courant = sut.get()

    # assert
    np.testing.assert_allclose(
        actual=max_courant,
        desired=np.maximum(
            np.maximum(abs(courant_field[0][1:, :]), abs(courant_field[0][:-1, :])),
            np.maximum(abs(courant_field[1][:, 1:]), abs(courant_field[1][:, :-1])),
        ),
    )
