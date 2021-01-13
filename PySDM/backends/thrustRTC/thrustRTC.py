"""
Created at 01.08.2019
"""

from PySDM.backends.thrustRTC.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.thrustRTC.impl._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.thrustRTC.impl._storage_methods import StorageMethods
from PySDM.backends.thrustRTC.impl._maths_methods import MathsMethods
from PySDM.backends.thrustRTC.impl._physics_methods import PhysicsMethods
from .storage.storage import Storage as ImportedStorage
from PySDM.backends.thrustRTC.storage.indexed_storage import IndexedStorage as ImportedIndexedStorage
from PySDM.backends.thrustRTC.random import Random as ImportedRandom


class ThrustRTC(
    AlgorithmicMethods,
    AlgorithmicStepMethods,
    StorageMethods,
    MathsMethods,
    PhysicsMethods,
):
    ENABLE = True
    Storage = ImportedStorage
    IndexedStorage = ImportedIndexedStorage
    Random = ImportedRandom

    def __init__(self):
        raise Exception("Backend is stateless.")

    @staticmethod
    def make_condensation_solver(coord='volume logarithm', adaptive=True, enable_drop_temperatures=False):
        return None
