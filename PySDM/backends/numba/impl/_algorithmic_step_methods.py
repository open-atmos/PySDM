"""
Created at 18.03.2020
"""

import numba
import numpy as np
from numba import float64, int64, void, prange, bool_

from PySDM.backends.numba import conf


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
    # @numba.njit(**conf.JIT_FLAGS)  # Note: in Numba 0.51 "np.dot() only supported on float and complex arrays"
    def cell_id_body(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin)

    @staticmethod
    def cell_id(cell_id, cell_origin, strides):
        return AlgorithmicStepMethods.cell_id_body(cell_id.data, cell_origin.data, strides.data)

    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in prange(length - 1):
            if is_first_in_pair[i]:
                data_out[i//2] = np.abs(data_in[idx[i]] - data_in[idx[i + 1]])

    @staticmethod
    @numba.njit(void(int64[:], bool_[:], int64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def find_pairs_body(cell_start, is_first_in_pair, cell_id, cell_idx, idx, length):
        for i in prange(length - 1):
            is_first_in_pair[i] = (
                    cell_id[idx[i]] == cell_id[idx[i + 1]] and
                    (i - cell_start[cell_idx[cell_id[idx[i]]]]) % 2 == 0
            )

    @staticmethod
    def find_pairs(cell_start, is_first_in_pair, cell_id, cell_idx, idx):
        return AlgorithmicStepMethods.find_pairs_body(
            cell_start.data, is_first_in_pair.data, cell_id.data, cell_idx.data, idx.data, len(idx))


    @staticmethod
    @numba.njit(void(float64[:], int64[:], bool_[:], int64[:], int64), **conf.JIT_FLAGS)
    def max_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in prange(length - 1):
            if is_first_in_pair[i]:
                data_out[i//2] = max(data_in[idx[i]], data_in[idx[i + 1]])

    @staticmethod
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        return AlgorithmicStepMethods.max_pair_body(data_out.data, data_in.data, is_first_in_pair.data, idx.data, length)

    @staticmethod
    @numba.njit(void(float64[:], float64[:], bool_[:], int64[:], int64), **conf.JIT_FLAGS)
    def sort_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in prange(length - 1):
            if is_first_in_pair[i]:
                if data_in[idx[i]] < data_in[idx[i + 1]]:
                    data_out[i], data_out[i + 1] = data_in[idx[i + 1]], data_in[idx[i]]
                else:
                    data_out[i], data_out[i + 1] = data_in[idx[i]], data_in[idx[i + 1]]

    @staticmethod
    def sort_pair(data_out, data_in, is_first_in_pair, idx, length):
        return AlgorithmicStepMethods.sort_pair_body(
            data_out.data, data_in.data, is_first_in_pair.data, idx.data, length)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def sort_within_pair_by_attr_body(idx, length, is_first_in_pair, attr):
        for i in prange(length - 1):
            if is_first_in_pair[i]:
                if attr[idx[i]] < attr[idx[i + 1]]:
                    idx[i], idx[i + 1] = idx[i + 1], idx[i]

    @staticmethod
    def sort_within_pair_by_attr(idx, length, is_first_in_pair, attr):
        AlgorithmicStepMethods.sort_within_pair_by_attr_body(
            idx.data, length, is_first_in_pair.indicator.data, attr.data)

    @staticmethod
    @numba.njit(void(float64[:], float64[:], bool_[:], int64[:], int64), **conf.JIT_FLAGS)
    def sum_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in prange(length - 1):
            if is_first_in_pair[i]:
                data_out[i//2] = (data_in[idx[i]] + data_in[idx[i + 1]])

    @staticmethod
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        return AlgorithmicStepMethods.sum_pair_body(data_out.data, data_in.data, is_first_in_pair.data, idx.data, length)
