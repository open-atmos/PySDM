"""
Created at 04.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64, boolean, prange
from PySDM.conf import NUMBA_PARALLEL


class MathsMethods:
    @staticmethod
    # @numba.njit([void(float64[:], float64[:]),
    #              void(float64[:, :], float64[:, :]),
    #              void(int64[:, :], int64[:, :]),
    #              void(float64[:, :], int64[:, :])],
    #             parallel=NUMBA_PARALLEL)
    def add(data_out, data_in):
        data_out[:] += data_in[:]

    @staticmethod
    # @numba.njit([float64(float64[:], int64[:], int64),
    #              int64(int64[:], int64[:], int64)])
    def amax(row, idx, length):
        result = np.amax(row[idx[:length]])
        return result

    @staticmethod
    # @numba.njit([float64(float64[:], int64[:], int64),
    #              int64(int64[:], int64[:], int64)])
    def amin(row, idx, length):
        result = np.amin(row[idx[:length]])
        return result

    @staticmethod
    @numba.njit(parallel=NUMBA_PARALLEL)
    def column_modulo(data, divisor):
        for d in range(len(divisor)):
            for i in prange(data.shape[0]):
                data[i, d] %= divisor[d]

    @staticmethod
    #@numba.njit()  # TODO
    def floor(data_out, data_in):
        data_out[:] = np.floor(data_in)

    @staticmethod
    #@numba.njit(void(float64[:]), parallel=NUMBA_PARALLEL)
    def floor_in_place(row):
        row[:] = np.floor(row)

    @staticmethod
    @numba.njit([void(float64[:], float64),
                 void(float64[:], float64[:]),
                 void(int64[:, :], int64)])
    def multiply(data, multiplier):
        data *= multiplier

    @staticmethod
    #@numba.njit()
    def subtract(data_out, data_in):
        data_out -= data_in

    @staticmethod
    @numba.njit(void(float64[:]))
    def urand(data):
        data[:] = np.random.uniform(0, 1, data.shape)
