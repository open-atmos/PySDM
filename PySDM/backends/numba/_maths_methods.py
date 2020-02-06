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
    @numba.njit([void(float64[:], float64[:]),
                 void(float64[:, :], float64[:, :]),
                 void(int64[:, :], int64[:, :]),
                 void(float64[:, :], int64[:, :])],
                **{**conf.JIT_FLAGS, **{'parallel': False}})
    def add(data_out, data_in):
        data_out[:] += data_in[:]

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
    @numba.njit(**conf.JIT_FLAGS)
    def column_modulo(data, divisor):
        for d in range(len(divisor)):
            for i in prange(data.shape[0]):
                data[i, d] %= divisor[d]

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def floor(data_out, data_in):
        data_out[:] = np.floor(data_in)

    @staticmethod
    @numba.njit(void(float64[:]), **conf.JIT_FLAGS)
    def floor_in_place(row):
        row[:] = np.floor(row)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def multiply(data_out, data_in, multiplier):
        data_out[:] = data_in * multiplier

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def multiply_in_place(data, multiplier):
        data *= multiplier

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def power(data, exponent):
        data[:] = np.power(data, exponent)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def subtract(data_out, data_in):
        data_out -= data_in

    @staticmethod
    @numba.njit(void(float64[:]), **conf.JIT_FLAGS)
    def urand(data):
        data[:] = np.random.uniform(0, 1, data.shape)
