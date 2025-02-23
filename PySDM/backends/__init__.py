"""
Backend classes: CPU=`PySDM.backends.numba.Numba`
and GPU=`PySDM.backends.thrust_rtc.ThrustRTC`
"""

import ctypes
import os
import sys
import warnings

from numba import cuda

from .numba import Numba


# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def _cuda_is_available():
    lib_names = ("libcuda.so", "libcuda.dylib", "cuda.dll")
    for libname in lib_names:
        try:
            cuda_lib = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        return False

    result = cuda_lib.cuInit(0)
    if result != 0:  # cuda.h: CUDA_SUCCESS = 0
        error_str = ctypes.c_char_p()
        cuda_lib.cuGetErrorString(result, ctypes.byref(error_str))
        # pylint: disable=no-member
        warnings.warn(
            f"CUDA library found but cuInit() failed (error code: {result};"
            f" message: {error_str.value.decode()})"
        )
        if "google.colab" in sys.modules:
            warnings.warn(
                "to use GPU on Colab set hardware accelerator to 'GPU' before session start"
                ' in the "Runtime :: Change runtime type :: Hardware accelerator" menu'
            )
        return False

    return True


if "CI" not in os.environ and (_cuda_is_available() or cuda.is_available()):
    from PySDM.backends.thrust_rtc import ThrustRTC
else:
    from .impl_thrust_rtc.test_helpers import flag

    flag.fakeThrustRTC = True

    import numpy as np

    from PySDM.backends.impl_common.random_common import (  # pylint: disable=ungrouped-imports
        RandomCommon,
    )
    from PySDM.backends.thrust_rtc import ThrustRTC  # pylint: disable=ungrouped-imports

    ThrustRTC.ENABLE = False

    class Random(RandomCommon):  # pylint: disable=too-few-public-methods
        def __init__(self, size, seed):
            super().__init__(size, seed)
            self.generator = np.random.default_rng(seed)

        def __call__(self, storage):
            # pylint: disable=unsupported-assignment-operation
            storage.data.ndarray[:] = self.generator.uniform(0, 1, storage.shape)

    ThrustRTC.Random = Random

CPU = Numba
"""
alias for Numba
"""

GPU = ThrustRTC
"""
alias for ThrustRTC
"""
