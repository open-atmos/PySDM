"""
Created at 24.07.2019
"""

from .numba.numba import Numba
from numba import cuda

if cuda.is_available():
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
