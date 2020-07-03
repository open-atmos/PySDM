"""
Created at 24.07.2019
"""

from ._methods import Methods
from ._algorithmic_methods import AlgorithmicMethods
from ._algorithmic_step_methods import AlgorithmicStepMethods
from ._storage_methods import StorageMethods
from ._maths_methods import MathsMethods
from ._physics_methods import PhysicsMethods
from .condensation_methods import CondensationMethods
from .storage import Storage as ImportedStorage
from .indexed_storage import IndexedStorage as ImportedIndexedStorage
from .random import Random as ImportedRandom


class Numba(
    Methods,
    AlgorithmicMethods,
    AlgorithmicStepMethods,
    StorageMethods,
    MathsMethods,
    PhysicsMethods,
    CondensationMethods
):
    Storage = ImportedStorage
    IndexedStorage = ImportedIndexedStorage
    Random = ImportedRandom
