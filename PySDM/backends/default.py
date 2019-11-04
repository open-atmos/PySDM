"""
Created at 31.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.backends.numba.numba import Numba
# from SDM.backends.thrustRTC import ThrustRTC
# from SDM.backends.pythran import Pythran

# TODO: backend::get_permuted_iterator + idx/length private to state


class Default(Numba):
    pass
