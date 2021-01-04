"""
Created at 04.11.2019
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
                 void(int64[:], int64[:]),
                 void(float64[:, :], int64[:, :]),
                 void(float64[:], float64)],
                **{**conf.JIT_FLAGS, **{'parallel': False}})
    def add(output, addend):
        output += addend

    @staticmethod
    @numba.njit(void(int64[:, :], int64[:]), **conf.JIT_FLAGS)
    def row_modulo(output, divisor):
        for d in range(len(divisor)):
            for i in prange(output.shape[1]):
                output[d, i] %= divisor[d]

    @staticmethod
    @numba.njit(void(float64[:]), **conf.JIT_FLAGS)
    def floor(output):
        output[:] = np.floor(output)

    @staticmethod
    @numba.njit(void(int64[:, :], float64[:, :]), **conf.JIT_FLAGS)
    def floor_out_of_place(output, input_data):
        output[:] = np.floor(input_data)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def multiply(output, multiplier):
        output *= multiplier

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def multiply_out_of_place(output, multiplicand, multiplier):
        output[:] = multiplicand * multiplier

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def power(output, exponent):
        output[:] = np.power(output, exponent)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def subtract(output, subtrahend):
        output[:] -= subtrahend[:]

    @staticmethod
    # @numba.njit(void(float64[:]), **conf.JIT_FLAGS)
    def urand(output, seed=None):
        np.random.seed(seed)
        output.data[:] = np.random.uniform(0, 1, output.shape)
