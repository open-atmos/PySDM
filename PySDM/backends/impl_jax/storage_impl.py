"""
Numba njit-ted basic arithmetics routines for CPU backend
"""

# import numba
# from PySDM.backends.impl_numba import conf
import jax


# @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
@jax.jit
def multiply(output, multiplier):
    output *= multiplier


@jax.jit
def divide_out_of_place(output, dividend, divisor):
    output = output.at[:].set(dividend / divisor)
