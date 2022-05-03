# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import numpy as np
import pytest

from PySDM.backends.impl_common.index import make_Index
from PySDM.backends.impl_common.indexed_storage import make_IndexedStorage
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.backends.impl_numba.methods.collisions_methods import pair_indices

from ...backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")


@pytest.mark.parametrize(
    "i, idx, is_first_in_pair, expected",
    [
        (0, (0, 1), (True, False), (0, 1)),
        (0, (1, 0), (True, False), (1, 0)),
        (0, (0, 1, 2), (False, True), (1, 2)),
    ],
)
def test_pair_indices(i, idx, is_first_in_pair, expected):
    # Arrange
    sut = pair_indices if "NUMBA_DISABLE_JIT" in os.environ else pair_indices.py_func

    # Act
    actual = sut(i, idx, is_first_in_pair)

    # Assert
    assert expected == actual


class TestAlgorithmicMethods:
    @staticmethod
    @pytest.mark.parametrize(
        "dt_left, cell_start, expected",
        [
            ((4, 5, 4.5, 0, 0), (0, 1, 2, 3, 4, 5), 3),
            ((4, 5, 4.5, 3, 0.1), (0, 1, 2, 3, 4, 5), 5),
        ],
    )
    # pylint: disable=redefined-outer-name
    def test_adaptive_sdm_end(backend_class, dt_left, cell_start, expected):
        # Arrange
        backend = backend_class()
        dt_left = backend.Storage.from_ndarray(np.asarray(dt_left))
        cell_start = backend.Storage.from_ndarray(np.asarray(cell_start))

        # Act
        actual = backend.adaptive_sdm_end(dt_left, cell_start)

        # Assert
        assert actual == expected

    @staticmethod
    @pytest.mark.parametrize(
        "gamma, idx, n, cell_id, dt_left, dt, dt_max, "
        "is_first_in_pair, "
        "expected_dt_left, expected_n_substep",
        (
            (
                (10.0,),
                (0, 1),
                (44, 44),
                (0, 0),
                (10.0,),
                10.0,
                10.0,
                (True, False),
                (9.0,),
                (1,),
            ),
            (
                (10.0,),
                (0, 1),
                (44, 44),
                (0, 0),
                (10.0,),
                10.0,
                0.1,
                (True, False),
                (9.9,),
                (1,),
            ),
            (
                (0.0,),
                (0, 1),
                (44, 44),
                (0, 0),
                (10.0,),
                10.0,
                10.0,
                (False, True),
                (0.0,),
                (1,),
            ),
            (
                (10.0,),
                (0, 1),
                (440, 44),
                (0, 0),
                (10.0,),
                10.0,
                10.0,
                (True, False),
                (0.0,),
                (1,),
            ),
            (
                (0.5, 6),
                (0, 1, 2, 3, 4),
                (44, 44, 22, 33, 11),
                (0, 0, 0, 1, 1),
                (10.0, 10),
                10.0,
                10.0,
                (True, False, False, True, False),
                (0.0, 5.0),
                (1, 1),
            ),
        ),
    )
    # pylint: disable=redefined-outer-name
    def test_adaptive_sdm_gamma(
        *,
        backend_class,
        gamma,
        idx,
        n,
        cell_id,
        dt_left,
        dt,
        dt_max,
        is_first_in_pair,
        expected_dt_left,
        expected_n_substep,
    ):
        # Arrange
        backend = backend_class()
        _gamma = backend.Storage.from_ndarray(np.asarray(gamma))
        _idx = make_Index(backend).from_ndarray(np.asarray(idx))
        _n = make_IndexedStorage(backend).from_ndarray(_idx, np.asarray(n))
        _cell_id = backend.Storage.from_ndarray(np.asarray(cell_id))
        _dt_left = backend.Storage.from_ndarray(np.asarray(dt_left))
        _is_first_in_pair = make_PairIndicator(backend)(len(n))
        _is_first_in_pair.indicator[:] = np.asarray(is_first_in_pair)
        _n_substep = backend.Storage.from_ndarray(np.zeros_like(dt_left, dtype=int))
        _dt_min = backend.Storage.from_ndarray(np.zeros_like(dt_left))
        dt_range = (np.nan, dt_max)

        # Act
        backend_class().adaptive_sdm_gamma(
            gamma=_gamma,
            n=_n,
            cell_id=_cell_id,
            dt_left=_dt_left,
            dt=dt,
            dt_range=dt_range,
            is_first_in_pair=_is_first_in_pair,
            stats_n_substep=_n_substep,
            stats_dt_min=_dt_min,
        )

        # Assert
        np.testing.assert_array_almost_equal(
            _dt_left.to_ndarray(), np.asarray(expected_dt_left)
        )
        expected_gamma = np.empty_like(np.asarray(gamma))
        for i in range(len(idx)):
            if is_first_in_pair[i]:
                expected_gamma[i // 2] = (
                    (dt - np.asarray(expected_dt_left[cell_id[i]]))
                    / dt
                    * np.asarray(gamma)[i // 2]
                )
        np.testing.assert_array_almost_equal(_gamma.to_ndarray(), expected_gamma)
        np.testing.assert_array_equal(_n_substep, np.asarray(expected_n_substep))
