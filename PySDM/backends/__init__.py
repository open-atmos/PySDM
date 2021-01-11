"""
Created at 24.07.2019
"""

from .numba.numba import Numba
import ctypes
import warnings
from numba import cuda


# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def cuda_is_available():
    lib_names = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
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
        warnings.warn(
            "CUDA library found but cuInit() failed (error code: %d; message: %s)" % (result, error_str.value.decode()))
        return False

    return True


if cuda_is_available() or cuda.is_available():
    from .thrustRTC.thrustRTC import ThrustRTC
else:
    from .thrustRTC.fakeThrustRTC import _flag

    _flag.fakeThrustRTC = True

    import numpy as np

    from .thrustRTC.thrustRTC import ThrustRTC
    ThrustRTC.ENABLE = False

    class Random:
        def __init__(self, size, seed=None):
            self.size = size
            seed = seed or np.random.randint(0, 2 * 16)
            self.generator = np.random.default_rng(seed)

        def __call__(self, storage):
            storage.data.ndarray[:] = self.generator.uniform(0, 1, storage.shape)

    ThrustRTC.Random = Random
   
CPU = Numba
GPU = ThrustRTC
