"""
Created at 20.04.2020
"""

import pytest
import os
import numpy as np
from PySDM.backends.numba.impl._algorithmic_methods import pair_indices
from PySDM.storages.index import make_Index
from PySDM.storages.pair_indicator import make_PairIndicator
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


@pytest.mark.parametrize("i, idx, is_first_in_pair, expected", [
    (0, (0, 1), (True, False), (0, 1)),
    (0, (1, 0), (True, False), (1, 0)),
    (0, (0, 1, 2), (False, True), (1, 2)),
])
def test_pair_indices(i, idx, is_first_in_pair, expected):
    # Arrange
    sut = pair_indices if 'NUMBA_DISABLE_JIT' in os.environ else pair_indices.py_func

    # Act
    actual = sut(i, idx, is_first_in_pair)

    # Assert
    assert expected == actual


class TestAlgorithmicMethods:
    @staticmethod
    @pytest.mark.parametrize("dt_left, cell_start, expected", [
        ((4, 5, 4.5, 0, 0), (0, 1, 2, 3, 4, 5), 3),
        ((4, 5, 4.5, 3, .1), (0, 1, 2, 3, 4, 5), 5)
    ])
    def test_adaptive_sdm_end(backend, dt_left, cell_start, expected):
        # Arrange
        dt_left = backend.Storage.from_ndarray(np.asarray(dt_left))
        cell_start = backend.Storage.from_ndarray(np.asarray(cell_start))

        # Act
        actual = backend.adaptive_sdm_end(dt_left, cell_start)

        # Assert
        assert actual == expected

    @staticmethod
    @pytest.mark.parametrize("gamma, idx, n, cell_id, dt_left, dt, dt_max, is_first_in_pair, expected", [
        ((10.,), (0, 1), (44, 44), (0, 0), (10.,), 10., 10., (True, False), (9.,)),
        ((10.,), (0, 1), (44, 44), (0, 0), (10.,), 10., .1, (True, False), (9.9,)),
        ((0.,), (0, 1), (44, 44), (0, 0), (10.,), 10., 10., (False, True), (0.,)),
        ((10.,), (0, 1), (440, 44), (0, 0), (10.,), 10., 10., (True, False), (0.,)),
        ((.5, 6), (0, 1, 2, 3, 4), (44, 44, 22, 33, 11), (0, 0, 0, 1, 1), (10., 10), 10., 10., (True, False, False, True, False), (0., 5.)),
    ])
    def test_adaptive_sdm_gamma(backend, gamma, idx, n, cell_id, dt_left, dt, dt_max, is_first_in_pair, expected):
        # Arrange
        _gamma = backend.Storage.from_ndarray(np.asarray(gamma))
        _idx = make_Index(backend).from_ndarray(np.asarray(idx))
        _n = backend.Storage.from_ndarray(np.asarray(n))
        _cell_id = backend.Storage.from_ndarray(np.asarray(cell_id))
        _dt_left = backend.Storage.from_ndarray(np.asarray(dt_left))
        _is_first_in_pair = make_PairIndicator(backend)(len(n))
        _is_first_in_pair.indicator[:] = np.asarray(is_first_in_pair)

        # Act
        backend.adaptive_sdm_gamma(_gamma, _idx, _n, _cell_id, _dt_left, dt, dt_max, _is_first_in_pair)

        # Assert
        np.testing.assert_array_almost_equal(_dt_left.to_ndarray(), np.asarray(expected))
        expected_gamma = (dt - np.asarray(expected)) / dt * np.asarray(gamma)
        np.testing.assert_array_almost_equal(_gamma.to_ndarray(), expected_gamma)
