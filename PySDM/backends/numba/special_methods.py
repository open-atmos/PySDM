"""
Created at 04.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64, boolean, prange
from PySDM.conf import NUMBA_PARALLEL


class SpecialMethods:
    @staticmethod
    @numba.njit(int64(int64[:], int64[:], int64))
    def remove_zeros(data, idx, length) -> int:
        new_length = 0
        for i in range(length):
            if data[idx[i]] == 0:
                idx[i] = len(idx)
            else:
                new_length += 1
        idx[:length].sort()
        return new_length

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

    # TODO comment
    @staticmethod
    @numba.jit()
    def compute_gamma(prob, rand):
        prob *= -1.
        prob[0::2] += rand
        prob[1::2] += rand
        prob[:] = np.floor(prob)
        prob *= -1.

    @staticmethod
    @numba.njit(boolean(int64[:]))
    def first_element_is_zero(arr):
        return arr[0] == 0

    @staticmethod
    def cell_id(cell_id, cell_origin, grid):
        # <TODO> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        domain = np.empty(tuple(grid))
        # </TODO> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        strides = np.array(domain.strides) / cell_origin.itemsize
        strides = strides.reshape(1, -1)  # transpose
        cell_id[:] = np.dot(strides, cell_origin.T)

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64[:], int64[:], int64))
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, sd_num):
        for i in range(sd_num - 1, -1, -1):  # reversed
            cell_start[cell_id[idx[i]]] = i
        cell_start[-1] = sd_num

        for i in range(sd_num - 1):
            is_first_in_pair[i] = (
                cell_id[idx[i]] == cell_id[idx[i+1]] and
                (i - cell_start[cell_id[idx[i]]]) % 2 == 0
            )