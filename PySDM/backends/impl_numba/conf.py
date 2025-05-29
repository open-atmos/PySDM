"""
default settings for Numba just-in-time compilation
"""

JIT_FLAGS = {
    "parallel": False,
    "fastmath": True,
    "error_model": "numpy",
    "cache": False,  # https://github.com/numba/numba/issues/2956
}
