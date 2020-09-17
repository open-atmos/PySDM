"""
Created at 24.07.2019
"""

from .numba.numba import Numba
from numba import cuda

if cuda.is_available():
    from .thrustRTC.thrustRTC import ThrustRTC
else:
    class ThrustRTC:
        ENABLE = False
   
CPU = Numba
GPU = ThrustRTC
