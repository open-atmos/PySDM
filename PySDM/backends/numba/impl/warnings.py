import numba
import sys
from PySDM.backends.numba import conf


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def warn(msg, file, context=None, return_value=None):
    with numba.objmode():
        print(msg, file=sys.stderr)
        print("\tfile:", file, file=sys.stderr)
        if context is not None:
            print("\tcontext:", file=sys.stderr)
            for var in context:
                print("\t\t", var, file=sys.stderr)
    return return_value