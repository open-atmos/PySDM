"""
Created at 24.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64, boolean

# TODO rename args

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
            raise NotImplementedError
        return data

    @staticmethod
    # @numba.njit()
    def from_ndarray(array):
        if str(array.dtype).startswith('int'):
            dtype = np.int64
        elif str(array.dtype).startswith('float'):
            dtype = np.float64
        else:
            raise NotImplementedError

        result = array.astype(dtype).copy()
        return result

    @staticmethod
    def write_row(array, i, row):
        array[i, :] = row

    @staticmethod
    def read_row(array, i):
        return array[i, :]

    # TODO idx -> self.idx?
    @staticmethod
    @numba.njit(void(int64[:], int64, int64))
    def shuffle(data, length, axis):
        idx = np.random.permutation(length)

        if axis == 0:
            data[:length] = data[idx[:length]]
        else:
            raise NotImplementedError

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                int64(int64[:], int64[:], int64)])
    def amin(row, idx, length):
        result = np.amin(row[idx[:length]])
        return result

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                int64(int64[:], int64[:], int64)])
    def amax(row, idx, length):
        result = np.amax(row[idx[:length]])
        return result

    @staticmethod
    # @numba.njit()
    def shape(data):
        return data.shape

    @staticmethod
    # @numba.njit()
    def dtype(data):
        return data.dtype

    @staticmethod
    @numba.njit(void(float64[:]))
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
    @numba.njit(void(int64[:], int64[:], int64, float64[:, :], float64[:, :], float64[:], int64[:]))
    def coalescence(n, idx, length, intensive, extensive, gamma, healthy):
        # TODO in segments
        for i in range(length // 2):
            j = 2 * i
            k = j + 1

            j = idx[j]
            k = idx[k]

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

    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64))
    def sum_pair(data_out, data_in, idx, length):
        for i in range(length // 2):
            data_out[i] = data_in[idx[2 * i]] + data_in[idx[2 * i + 1]]

    @staticmethod
    @numba.njit(void(float64[:], int64[:], int64[:], int64))
    def max_pair(data_out, data_in, idx, length):
        for i in range(length // 2):
            data_out[i] = max(data_in[idx[2 * i]], data_in[idx[2 * i + 1]])

    @staticmethod
    @numba.njit([void(float64[:], float64),
                 void(float64[:], float64[:])])
    def multiply(data, multiplier):
        data *= multiplier

    @staticmethod
    @numba.njit(void(float64[:], float64[:]))
    def sum(data_out, data_in):
        data_out[:] = data_out + data_in

    @staticmethod
    @numba.njit(void(float64[:]))
    def floor(row):
        row[:] = np.floor(row)

    @staticmethod
    # @numba.njit()
    def to_ndarray(data):
        return data.copy()

    @staticmethod
    @numba.njit(boolean(int64[:]))
    def first_element_is_zero(arr):
        return arr[0] == 0
