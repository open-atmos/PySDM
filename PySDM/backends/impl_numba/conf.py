"""
default settings for Numba just-in-time compilation
"""
import os
import warnings

import numba

JIT_FLAGS = {
    "parallel": True,
    "fastmath": True,
    "error_model": "numpy",
    "cache": False,  # https://github.com/numba/numba/issues/2956
}

try:
    numba.parfors.parfor.ensure_parallel_support()
except numba.core.errors.UnsupportedParforsError:
    if "CI" not in os.environ:
        warnings.warn("Numba version used does not support parallel for (32 bits?)")
    JIT_FLAGS["parallel"] = False
