"""
Backend classes: `CPU`=`PySDM.backends.numba.numba.Numba`
and `GPU`=`PySDM.backends.thrustRTC.thrustRTC.ThrustRTC`
"""
from .numba.numba import Numba
import ctypes
import warnings
from numba import cuda
import sys


# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def _cuda_is_available():
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
        if 'google.colab' in sys.modules:
            warnings.warn("to use GPU on Colab set hardware accelerator to 'GPU' before session start"
                         'in the "Runtime :: Change runtime type :: Hardware accelerator" menu')
        return False

    return True


if _cuda_is_available() or cuda.is_available():
    from .thrustRTC.thrustRTC import ThrustRTC
else:
    from .thrustRTC.test_helpers import _flag

    _flag.fakeThrustRTC = True

    import numpy as np

    from .thrustRTC.thrustRTC import ThrustRTC
    ThrustRTC.ENABLE = False

    class Random:
        def __init__(self, size, seed):
            self.size = size
            self.generator = np.random.default_rng(seed)

        def __call__(self, storage):
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
