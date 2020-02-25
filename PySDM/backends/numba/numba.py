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
from PySDM.backends.numba.condensation_methods import CondensationMethods


class Numba(StorageMethods, MathsMethods, PhysicsMethods, SpecialMethods, CondensationMethods):
    @staticmethod
    def num_threads():
        return numba.config.NUMBA_NUM_THREADS
