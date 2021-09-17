import numpy as np
import pytest
from PySDM.physics import Formulae

# noinspection PyUnresolvedReferences
from ...backends_fixture import backend as Backend

@pytest.mark.parametrize("min_x, max_x, value, expected", [
    (0, 1, .5, 1),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (0, 1, -.5, 0),
    (0, 1, 1.5, 0),
])
def test_moments_range(Backend, min_x, max_x, value, expected):
    # Arrange
    backend = Backend(Formulae())
    arr = lambda x: backend.Storage.from_ndarray(np.asarray((x,)))

    moment_0 = arr(0)
    moments = backend.Storage.from_ndarray(np.full((1,1), 0))
    n = arr(1)
    attr_data = arr(0)
    cell_id = arr(0)
    idx = arr(0)
    length = 1
    ranks = arr(0)
    x_attr = arr(value)
    weighting_attribute = arr(0)
    weighting_rank = 0

    # Act
    backend.moments(
        moment_0=moment_0, moments=moments, n=n, attr_data=attr_data,
        cell_id=cell_id, idx=idx, length=length,
        ranks=ranks, min_x=min_x, max_x=max_x, x_attr=x_attr, weighting_attribute=weighting_attribute,
        weighting_rank=weighting_rank
    )

    # Assert
    assert moment_0.to_ndarray()[:] == moments.to_ndarray()[:] == expected