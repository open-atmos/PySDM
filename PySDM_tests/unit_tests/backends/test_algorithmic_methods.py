"""
Created at 20.04.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.backends.__parametrisation__ import \
    number_float, number, \
    shape_full, shape_1d, shape_2d, \
    natural_length, length, \
    order, pairs, \
    dtype_full, dtype, dtype_mixed
from PySDM_tests.unit_tests.backends.__parametrisation__ import backends
from .utils import universal_test


@pytest.mark.parametrize('sut', backends)
class TestAlgorithmicMethods:

    @staticmethod
    def test_calculate_displacement(sut):
        assert False

    @staticmethod
    def test_coalescence(sut):
        assert False

    @staticmethod
    def test_compute_gamma(sut):
        assert False

    @staticmethod
    def test_condensation(sut):
        assert False

    @staticmethod
    def test_flag_precipitated(sut):
        assert False

    @staticmethod
    def test_make_cell_caretaker(sut):
        assert False

    @staticmethod
    def test_moments(sut):
        assert False

    @staticmethod
    def test_normalize(sut):
        assert False

    @staticmethod
    @pytest.mark.parametrize('data_ndarray', [
        np.array([0] * 87),
        np.array([1, 0, 1, 0, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 1, 0, 1, 0]),
        np.array([1] * 87)
    ])
    def test_remove_zeros(sut, data_ndarray, length, order):
        params = [{'name': "data",
                   'details': {'array': data_ndarray}},
                  {'name': "idx",
                   'details': {'shape': (data_ndarray.shape[0], ), 'order': order},
                   'checking': ['sorted', 'length_valid']},
                  {'name': "length",
                   'details': {'shape': (data_ndarray.shape[0], ), 'length': length}}
                  ]
        universal_test("remove_zeros", sut, params)


