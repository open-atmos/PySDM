# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numba
import numpy as np
import pytest

from PySDM.backends.impl_common.pair_indicator import make_PairIndicator

from ...backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")


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
            marks=pytest.mark.xfail(
                strict=numba.config.DISABLE_JIT
            ),  # pylint: disable=no-member
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
def test_sum_pair_out_of_bounds(
    backend_class, _data_in, _data_out, _is_first_in_pair, _idx
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
    backend.sum_pair(data_out, data_in, is_first_in_pair, idx)

    # Assert
