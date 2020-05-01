"""
Created at 18.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from . import conf
import numba
from numba import float64, int64, void, prange
import numpy as np


class AlgorithmicStepMethods:

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                 int64(int64[:], int64[:], int64)], **conf.JIT_FLAGS)
    def amax(row, idx, length):
        result = np.amax(row[idx[:length]])
        return result

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                 int64(int64[:], int64[:], int64)], **conf.JIT_FLAGS)
    def amin(row, idx, length):
        result = np.amin(row[idx[:length]])
        return result

    @staticmethod
    # @numba.njit(**conf.JIT_FLAGS)  # TODO: "np.dot() only supported on float and complex arrays"
    def cell_id(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin.T)

    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = np.abs(data_in[idx[i]] - data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, length):
        for i in prange(length - 1):
            is_first_in_pair[i] = (
                cell_id[idx[i]] == cell_id[idx[i+1]] and
                (i - cell_start[cell_id[idx[i]]]) % 2 == 0
            )

    @staticmethod
    @numba.njit(void(float64[:], int64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = max(data_in[idx[i]], data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = (data_in[idx[i]] + data_in[idx[i + 1]]) if is_first_in_pair[i] else 0
