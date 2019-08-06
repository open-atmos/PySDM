"""
Created at 24.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64


# TODO backend.storage overrides __getitem__


class Numba:
    storage = np.ndarray

    @staticmethod
    # @numba.njit()
    def array(shape, dtype):
        if dtype is float:
            data = np.full(shape, np.nan, dtype=np.float64)
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
    # @numba.njit()
    def stack_2d(data, array):
        data[:] = np.concatenate((data, array), axis=0)

    @staticmethod
    def get_item(array, item):
        return array[item]

    # TODO idx as input
    @staticmethod
    @numba.njit(void(int64[:], int64, int64))
    def shuffle(data, length, axis):
        idx = np.random.permutation(length)

        if axis == 0:
            data[:length] = data[idx[:length]]
        else:
            raise NotImplementedError

    @staticmethod
    # @numba.njit(void(int64[:], float64[:], int64))
    def argsort(idx, data, length):
        idx[0:length] = data[0:length].argsort()

    @staticmethod
    # @numba.njit(void(int64[:], float64[:], int64))
    def stable_argsort(idx: np.ndarray, data: np.ndarray, length: int):
        idx[:length] = data[:length].argsort(kind='stable')

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                int64(int64[:], int64[:], int64)])
    def amin(data, idx, length):
        result = np.amin(data[idx[:length]])
        return result

    @staticmethod
    @numba.njit([float64(float64[:], int64[:], int64),
                int64(int64[:], int64[:], int64)])
    def amax(data, idx, length):
        result = np.amax(data[idx[:length]])
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
    @numba.njit(void(int64[:], int64[:], int64, float64[:, :], float64[:]))
    def extensive_attr_coalescence(n, idx, length, data, gamma):
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
                data[:, k] += g * data[:, j]
            else:  # new_n == 0
                data[:, j] = g * data[:, j] + data[:, k]
                data[:, k] = data[:, j]

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64, float64[:]))
    def n_coalescence(n, idx, length, gamma):
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
            else:  # new_n == 0
                n[j] = n[k] // 2
                n[k] = n[k] - n[j]

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
    def floor(data):
        data[:] = np.floor(data)

    @staticmethod
    # @numba.njit()
    def to_ndarray(data):
        return data




