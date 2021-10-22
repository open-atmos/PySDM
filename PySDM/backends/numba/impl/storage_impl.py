import numpy as np
import numba
from PySDM.backends.numba import conf


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def add(output, addend):
    output += addend


@numba.njit(**conf.JIT_FLAGS)
def amin(data):
    return np.amin(data)


@numba.njit(**conf.JIT_FLAGS)
def row_modulo(output, divisor):
    for d in range(output.shape[0]):
        for i in numba.prange(output.shape[1]):
            output[d, i] %= divisor[d]


@numba.njit(**conf.JIT_FLAGS)
def floor(output):
    output[:] = np.floor(output)


@numba.njit(**conf.JIT_FLAGS)
def floor_out_of_place(output, input_data):
    output[:] = np.floor(input_data)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def multiply(output, multiplier):
    output *= multiplier


@numba.njit(**conf.JIT_FLAGS)
def multiply_out_of_place(output, multiplicand, multiplier):
    output[:] = multiplicand * multiplier


@numba.njit(**conf.JIT_FLAGS)
def divide_out_of_place(output, dividend, divisor):
    output[:] = dividend / divisor

@numba.njit(**conf.JIT_FLAGS)
def sum_out_of_place(output, a, b):
    output[:] = a + b

@numba.njit(**conf.JIT_FLAGS)
def power(output, exponent):
    output[:] = np.power(output, exponent)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def subtract(output, subtrahend):
    output[:] -= subtrahend[:]


# @numba.njit(void(f8[:]), **conf.JIT_FLAGS)
def urand(output):
    output.data[:] = np.random.uniform(0, 1, output.shape)
