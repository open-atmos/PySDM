"""
Created at 24.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numba
from PySDM.backends.numba._storage_methods import StorageMethods
from PySDM.backends.numba._maths_methods import MathsMethods
from PySDM.backends.numba._physics_methods import PhysicsMethods
from PySDM.backends.numba._special_methods import SpecialMethods


class Numba(StorageMethods, MathsMethods, PhysicsMethods, SpecialMethods):
    @staticmethod
    def num_threads():
        return numba.config.NUMBA_NUM_THREADS
