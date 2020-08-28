"""
Created at 24.07.2019
"""

from .numba.numba import Numba
from .numba.numba import Numba as CPU

try:
    from .thrustRTC.thrustRTC import ThrustRTC
    from .thrustRTC.thrustRTC import ThrustRTC as GPU
except ImportError:
    class ThrustRTC:
        ENABLE = False
