"""
Created at 04.11.2019
"""

from numba.np.ufunc.parallel import _launch_threads
from numba.core.errors import NumbaWarning

try:
    _launch_threads()
except (NumbaWarning):
    pass
