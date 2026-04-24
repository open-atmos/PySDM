# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import numpy as np
import pytest

from PySDM.backends import CPU
from PySDM.backends.impl_common.index import make_Index
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator


class TestPairMethods:
    @staticmethod
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

        sut = (
            backend._sum_pair_body
            if "NUMBA_DISABLE_JIT" in os.environ
            else backend._sum_pair_body.py_func
        )
        # Act
        sut(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

        # Assert

    @staticmethod
    @pytest.mark.parametrize(
        "_data_in, _data_out, _idx",
        (
            pytest.param(
                [44.0, 666.0],
                [
                    0,
                ],
                [0, 1],
            ),
        ),
    )
    def test_sum_pair(_data_in, _data_out, _idx, backend_instance_with_jax):
        # Arrange
        backend = backend_instance_with_jax

        data_out = backend.Storage.from_ndarray(np.asarray(_data_out))
        data_in = backend.Storage.from_ndarray(np.asarray(_data_in))

        is_first_in_pair = make_PairIndicator(backend)(len(_data_in))
        is_first_in_pair.indicator = backend.Storage.from_ndarray(
            np.asarray(
                [True, False],
            )
        )

        idx = backend.Storage.from_ndarray(np.asarray(_idx))

        # Act
        backend.sum_pair(data_out, data_in, is_first_in_pair, idx)

        # Assert
        np.testing.assert_array_equal(data_out.to_ndarray(), [44.0 + 666.0])

    @staticmethod
    @pytest.mark.parametrize("length", (1, 2, 3, 4))
    def test_find_pairs_length(backend_instance_with_jax, length):
        # arrange
        backend = backend_instance_with_jax
        n_sd = 4

        cell_start = backend.Storage.from_ndarray(np.asarray([0, 0, 0, 0]))
        cell_id = backend.Storage.from_ndarray(np.asarray([0, 0, 0, 0]))
        cell_idx = backend.Storage.from_ndarray(np.asarray([0, 1, 2, 3]))
        is_first_in_pair = make_PairIndicator(backend)(n_sd)
        is_first_in_pair.indicator = backend.Storage.from_ndarray(
            np.asarray([True] * n_sd)
        )
        idx = make_Index(backend).identity_index(n_sd)

        # act
        idx.length = length
        backend.find_pairs(cell_start, is_first_in_pair, cell_id, cell_idx, idx)

        # assert
        assert not is_first_in_pair.indicator.to_ndarray()[length - 1]

    @staticmethod
    @pytest.mark.parametrize(
        "_attr, _idx, _expected_idx, _is_first_in_pair",
        (
            pytest.param(
                [44.0, 666.0],
                [0, 1],
                [1, 0],
                [True, False],
            ),
            pytest.param(
                [100.0, 10.0, 10.0, 100.0],
                [0, 1, 2, 3],
                [0, 1, 3, 2],
                [True, False, True, False],
            ),
            pytest.param(
                [5, 6, 7],
                [0, 1, 2],
                [0, 2, 1],
                [False, True, False],
            ),
            pytest.param(
                [100.0, 10.0, 10.0, 100.0, 10.0, 100.0],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 3, 2, 5, 4],
                [True, False, True, False, True, False],
            ),
            pytest.param(
                [100.0, 10.0, 10.0, 100.0],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [False, False, False, False],
            ),
            pytest.param(
                [100.0, 10.0, 10.0, 100.0],
                [1, 0, 3, 2],
                [0, 1, 3, 2],
                [True, False, True, False],
            ),
        ),
    )
    def test_sort_within_pair_by_attr(
        _attr, _idx, _expected_idx, _is_first_in_pair, backend_instance_with_jax
    ):
        # Arrange
        backend = backend_instance_with_jax

        attr = backend.Storage.from_ndarray(np.asarray(_attr))

        is_first_in_pair = make_PairIndicator(backend)(len(_attr))
        is_first_in_pair.indicator = backend.Storage.from_ndarray(
            np.asarray(_is_first_in_pair)
        )

        idx = backend.Storage.from_ndarray(np.asarray(_idx))

        # Act
        backend.sort_within_pair_by_attr(idx, is_first_in_pair, attr)

        # Assert
        np.testing.assert_array_equal(idx.to_ndarray(), _expected_idx)
