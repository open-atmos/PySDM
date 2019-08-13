"""
Created at 31.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.backends.numba import Numba
# from SDM.backends.thrustRTC import ThrustRTC
# from SDM.backends.pythran import Pythran

# TODO backend.storage overrides __getitem__
# TODO methods almost always with idx&length?


class Default(Numba):
    pass
