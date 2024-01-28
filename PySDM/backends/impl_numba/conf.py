"""
default settings for Numba just-in-time compilation
"""

import os
import platform
import warnings

import numba

JIT_FLAGS = {
    "parallel": True,
    "fastmath": True,
    "error_model": "numpy",
    "cache": False,  # https://github.com/numba/numba/issues/2956
}

if platform.machine() == "arm64":
    warnings.warn(
        "Disabling Numba threading due to ARM64 CPU (atomics do not work yet)"
    )
    JIT_FLAGS["parallel"] = False  # TODO #1183 - atomics don't work on ARM64!

try:
    numba.parfors.parfor.ensure_parallel_support()
except numba.core.errors.UnsupportedParforsError:
    if "CI" not in os.environ:
        warnings.warn("Numba version used does not support parallel for (32 bits?)")
    JIT_FLAGS["parallel"] = False
