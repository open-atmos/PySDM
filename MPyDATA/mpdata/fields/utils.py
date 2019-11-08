"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numba


@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def is_integral(n):
    return int(n * 2.) % 2 == 0


@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def is_fractional(n):
    return int(n * 2.) % 2 == 1
