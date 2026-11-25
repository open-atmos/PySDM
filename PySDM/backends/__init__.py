"""
Number-crunching backends
"""

from functools import partial

from . import numba as _numba
from . import jax as _jax
from .thrust_rtc import ThrustRTC

# for pdoc
CPU = None
GPU = None
JAX = None
Numba = _numba.Numba
Jax = _jax.Jax


_BACKEND_CACHE = {}


def _cached_backend(formulae=None, n_dims=0, backend_class=None, **kwargs):
    key = (
        backend_class.__name__
        + ":"
        + str(n_dims)
        + ":"
        + str(formulae)
        + ":"
        + str(kwargs)
    )
    if key not in _BACKEND_CACHE:
        _BACKEND_CACHE[key] = backend_class(formulae=formulae, n_dims=n_dims, **kwargs)
    return _BACKEND_CACHE[key]


CPU = partial(_cached_backend, backend_class=Numba)
""" returns a cached instance of the Numba backend (cache key including formulae parameters) """

GPU = partial(_cached_backend, backend_class=ThrustRTC)
""" returns a cached instance of the ThrustRTC backend (cache key including formulae parameters) """

JAX = partial(_cached_backend, backend_class=Jax)
