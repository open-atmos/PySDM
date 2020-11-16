"""
Created at 24.07.2019
"""

from numba.core.errors import NumbaWarning
imports_done = False
try: # python -We (warnings for incompatible versions of TBB which is optional) 
    from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
    from PySDM.backends.numba.impl._algorithmic_step_methods import AlgorithmicStepMethods
    from PySDM.backends.numba.impl._storage_methods import StorageMethods
    from PySDM.backends.numba.impl._maths_methods import MathsMethods
    from PySDM.backends.numba.impl._physics_methods import PhysicsMethods
    from PySDM.backends.numba.impl.condensation_methods import CondensationMethods
    from PySDM.backends.numba.storage.storage import Storage as ImportedStorage
    from PySDM.backends.numba.storage.indexed_storage import IndexedStorage as ImportedIndexedStorage
    from PySDM.backends.numba.random import Random as ImportedRandom
except (NumbaWarning):
    pass
else:
    imports_done = True
finally:
    assert imports_done

class Numba(
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

    def __init__(self):
        raise Exception("Backend is stateless.")
