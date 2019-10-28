"""
Created at 24.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64, boolean, prange
from SDM.conf import NUMBA_PARALLEL


class Numba:
    storage = np.ndarray

    @staticmethod
    # @numba.njit()
    def array(shape, dtype):
        if dtype is float:
            data = np.full(shape, -1., dtype=np.float64)
        elif dtype is int:
            data = np.full(shape, -1, dtype=np.int64)
        else:
            raise NotImplementedError()
        return data

    @staticmethod
    # @numba.njit()
    def from_ndarray(array):
        if str(array.dtype).startswith('int'):
            dtype = np.int64
        elif str(array.dtype).startswith('float'):
            dtype = np.float64
        else:
            raise NotImplementedError()

        result = array.astype(dtype).copy()
        return result

    @staticmethod
    def write_row(array, i, row):
        array[i, :] = row

    @staticmethod
    def read_row(array, i):
        return array[i, :]

    @staticmethod
    def stable_sort(idx, keys, length):
        idx[:length] = keys[idx[:length]].argsort(kind='stable')

    # TODO idx -> self.idx?
    @staticmethod
    @numba.njit(void(int64[:], int64, int64), parallel=NUMBA_PARALLEL)
    def shuffle(data, length, axis):
        idx = np.random.permutation(length)

        if axis == 0:
            data[:length] = data[idx[:length]]
        else:
            raise NotImplementedError()

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                 int64(int64[:], int64[:], int64)], parallel=NUMBA_PARALLEL)
    def amin(row, idx, length):
        result = np.amin(row[idx[:length]])
        return result

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                 int64(int64[:], int64[:], int64)], parallel=NUMBA_PARALLEL)
    def amax(row, idx, length):
        result = np.amax(row[idx[:length]])
        return result

    @staticmethod
    # @numba.njit()
    def shape(data):
        return data.shape

    @staticmethod
    def cell_id(cell_id, cell_origin, grid):
        # <TODO> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        domain = np.empty(tuple(grid))
        # </TODO> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        strides = np.array(domain.strides) / cell_origin.itemsize
        strides = strides.reshape(1, -1)  # transpose
        print("strides", strides)
        cell_id[:] = np.dot(strides, cell_origin.T)

    @staticmethod
    # @numba.njit()
    def dtype(data):
        return data.dtype

    @staticmethod
    @numba.njit(void(float64[:]), parallel=NUMBA_PARALLEL)
    def urand(data):
        data[:] = np.random.uniform(0, 1, data.shape)

    @staticmethod
    @numba.njit(int64(int64[:], int64[:], int64))
    def remove_zeros(data, idx, length) -> int:
        result = 0
        for i in range(length):
            if data[idx[i]] == 0:
                idx[i] = len(idx)
            else:
                result += 1
        idx[:length].sort()
        return result

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64, float64[:, :], float64[:, :], float64[:], int64[:]),
                parallel=NUMBA_PARALLEL)
    def coalescence(n, idx, length, intensive, extensive, gamma, healthy):
        for i in prange(length - 1):
            if gamma[i] == 0:
                continue

            j = idx[i]
            k = idx[i + 1]

            if n[j] < n[k]:
                j, k = k, j
            g = min(gamma[i], n[j] // n[k])
            if g == 0:
                continue

            new_n = n[j] - g * n[k]
            if new_n > 0:
                n[j] = new_n
                extensive[:, k] += g * extensive[:, j]
            else:  # new_n == 0
                n[j] = n[k] // 2
                n[k] = n[k] - n[j]
                extensive[:, j] = g * extensive[:, j] + extensive[:, k]
                extensive[:, k] = extensive[:, j]
            if n[k] == 0 or n[j] == 0:
                healthy[0] = 0

    # TODO: silently assumes that data_out is not permuted (i.e. not part of state)
    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), parallel=NUMBA_PARALLEL)
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        #        for i in prange(length // 2):
        #            data_out[i] = data_in[idx[2 * i]] + data_in[idx[2 * i + 1]]
        for i in prange(length - 1):
            data_out[i] = (data_in[idx[i]] + data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    # TODO: ditto
    @staticmethod
    @numba.njit(void(float64[:], int64[:], int64[:], int64[:], int64), parallel=NUMBA_PARALLEL)
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        # for i in prange(length // 2):
        #    data_out[i] = max(data_in[idx[2 * i]], data_in[idx[2 * i + 1]])
        for i in prange(length - 1):
            data_out[i] = max(data_in[idx[i]], data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    @staticmethod
    @numba.njit([void(float64[:], float64),
                 void(float64[:], float64[:]),
                 void(int64[:, :], int64)])  # TODO add subtract
    def multiply(data, multiplier):
        data *= multiplier

    # TODO add
    @staticmethod
    @numba.njit([void(float64[:], float64[:]),
                 void(float64[:, :], float64[:, :]),
                 void(int64[:, :], int64[:, :]),
                 void(float64[:, :], int64[:, :])],
                parallel=NUMBA_PARALLEL)
    def sum(data_out, data_in):
        data_out[:] = data_out + data_in

    @staticmethod
    @numba.njit(void(float64[:]), parallel=NUMBA_PARALLEL)
    def floor(row):
        row[:] = np.floor(row)

    # TODO
    @staticmethod
    @numba.njit()  # TODO
    def floor2(data_out, data_in):
        data_out[:] = np.floor(data_in)

    @staticmethod
    # @numba.njit()
    def to_ndarray(data):
        return data.copy()

    @staticmethod
    @numba.njit(boolean(int64[:]))
    def first_element_is_zero(arr):
        return arr[0] == 0
