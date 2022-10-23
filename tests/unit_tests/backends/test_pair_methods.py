# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.backends import CPU
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator


@pytest.mark.parametrize(
    "_data_in, _data_out, _is_first_in_pair, _idx",
    (
        pytest.param(
            [44.0, 666.0],
            [
                0,
            ],
            [True, True],
            [0, 1],
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            [44.0, 666.0],
            [
                0,
            ],
            [True, False],
            [0, 1],
        ),
    ),
)
# pylint: disable=redefined-outer-name
def test_sum_pair_body_out_of_bounds(
    _data_in, _data_out, _is_first_in_pair, _idx, backend_class=CPU
):
    # Arrange
    backend = backend_class()

    data_out = backend.Storage.from_ndarray(np.asarray(_data_out))
    data_in = backend.Storage.from_ndarray(np.asarray(_data_in))

    is_first_in_pair = make_PairIndicator(backend)(len(_is_first_in_pair))
    is_first_in_pair.indicator = backend.Storage.from_ndarray(
        np.asarray(_is_first_in_pair)
    )

    idx = backend.Storage.from_ndarray(np.asarray(_idx))

    # Act
    backend.sum_pair_body.py_func(
        data_out.data, data_in.data, is_first_in_pair.indicator.data, idx.data, len(idx)
    )

    # Assert
