"""
Created at 24.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .default import Default
from .numba.numba import Numba
import os
if os.environ.get('TRAVIS') != 'true':
    from .thrustRTC.thrustRTC import ThrustRTC
