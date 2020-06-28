"""
Created at 24.07.2019
"""

from .default import Default
from .numba.numba import Numba
import os
try:
    from .thrustRTC.thrustRTC import ThrustRTC
except ImportError:
    class ThrustRTC:
        ENABLE = False
