"""
Created at 24.07.2019
"""

from .numba.numba import Numba

try:
    from .thrustRTC.thrustRTC import ThrustRTC
except ImportError:
    class ThrustRTC:
        ENABLE = False
   
CPU = Numba
GPU = ThrustRTC
