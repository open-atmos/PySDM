"""
warning reporting logic for use whithin Numba njitted code (printing to standard
 error using numba.objmode() allowing to capture the output from Python tests
"""

import sys

import numba

from PySDM.backends.impl_numba import conf


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def warn(msg, file, context=None, return_value=None):
    with numba.objmode():
        print(msg, file=sys.stderr)
        print("\tfile:", file, file=sys.stderr)
        if context is not None:
            print("\tcontext:", file=sys.stderr)
            for var in context:
                print("\t\t", var, file=sys.stderr)
    return return_value
