"""
Created at 24.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numba
from PySDM.backends.numba._methods import Methods
from PySDM.backends.numba._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.numba._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.numba._storage_methods import StorageMethods
from PySDM.backends.numba._maths_methods import MathsMethods
from PySDM.backends.numba._physics_methods import PhysicsMethods
from PySDM.backends.numba.condensation_methods import CondensationMethods


class Numba(
    Methods,
    AlgorithmicMethods,
    AlgorithmicStepMethods,
    StorageMethods,
    MathsMethods,
    PhysicsMethods,
    CondensationMethods
):
    @staticmethod
    def num_threads():
        return numba.config.NUMBA_NUM_THREADS
