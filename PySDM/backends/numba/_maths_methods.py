"""
Created at 04.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64, prange
from PySDM.backends.numba import conf


class MathsMethods:
    @staticmethod
    # @numba.njit([void(float64[:], float64[:]),
    #              void(float64[:, :], float64[:, :]),
    #              void(int64[:, :], int64[:, :]),
    #              void(float64[:, :], int64[:, :])],
    #             parallel=conf.NUMBA_PARALLEL, fastmath=conf.NUMBA_FASTMATH)
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
    #              int64(int64[:], int64[:], int64)], fastmath=conf.NUMBA_FASTMATH)
    def amin(row, idx, length):
        result = np.amin(row[idx[:length]])
        return result

    @staticmethod
    @numba.njit(parallel=conf.NUMBA_PARALLEL, fastmath=conf.NUMBA_FASTMATH)
    def column_modulo(data, divisor):
        for d in range(len(divisor)):
            for i in prange(data.shape[0]):
                data[i, d] %= divisor[d]

    @staticmethod
    # @numba.njit(fastmath=conf.NUMBA_FASTMATH)  # TODO
    def floor(data_out, data_in):
        data_out[:] = np.floor(data_in)

    @staticmethod
    #@numba.njit(void(float64[:]), parallel=conf.NUMBA_PARALLEL, fastmath=conf.NUMBA_FASTMATH)
    def floor_in_place(row):
        row[:] = np.floor(row)

    @staticmethod
    # @numba.njit(fastmath=conf.NUMBA_FASTMATH)
    def multiply(data_out, data_in, multiplier):
        data_out[:] = data_in * multiplier

    @staticmethod
    # @numba.njit(fastmath=conf.NUMBA_FASTMATH)
    def multiply_in_place(data, multiplier):
        data *= multiplier

    @staticmethod
    def power(data, exponent):
        data[:] = np.power(data, exponent)

    @staticmethod
    #@numba.njit(fastmath=conf.NUMBA_FASTMATH)
    def subtract(data_out, data_in):
        data_out -= data_in

    @staticmethod
    @numba.njit(void(float64[:]), fastmath=conf.NUMBA_FASTMATH)
    def urand(data):
        data[:] = np.random.uniform(0, 1, data.shape)
