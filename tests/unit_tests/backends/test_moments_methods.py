# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae


@pytest.mark.parametrize(
    "min_x, max_x, value, expected",
    [
        (0, 1, 0.5, 1),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (0, 1, -0.5, 0),
        (0, 1, 1.5, 0),
    ],
)
def test_moments_range(backend_class, min_x, max_x, value, expected):
    # Arrange
    backend = backend_class(Formulae())

    def arr(x):
        return backend.Storage.from_ndarray(np.asarray((x,)))

    moment_0 = arr(0.0)
    moments = backend.Storage.from_ndarray(np.full((1, 1), 0.0))

    kw_args = {
        "multiplicity": arr(1),
        "attr_data": arr(0),
        "cell_id": arr(0),
        "idx": arr(0),
        "length": 1,
        "ranks": arr(0),
        "x_attr": arr(value),
        "weighting_attribute": arr(0),
        "weighting_rank": 0,
        "skip_division_by_m0": False,
    }

    # Act
    backend.moments(
        moment_0=moment_0, moments=moments, min_x=min_x, max_x=max_x, **kw_args
    )

    # Assert
    assert moment_0.to_ndarray()[:] == moments.to_ndarray()[:] == expected
