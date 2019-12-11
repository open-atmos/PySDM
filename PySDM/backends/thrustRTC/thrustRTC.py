"""
Created at 01.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.backends.numba.numba import Numba
from PySDM.conf import TRTC


if not TRTC:
    class ThrustRTC(Numba):
        pass
else:
    from ._storage_methods import StorageMethods
    from ._maths_methods import MathsMethods
    from ._special_methods import SpecialMethods

    class ThrustRTC(StorageMethods, MathsMethods, SpecialMethods):
        pass
