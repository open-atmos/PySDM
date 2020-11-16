"""
Created at 24.07.2019
"""

from numba.core.errors import NumbaWarning # python -We (warnings for incompatible versions of TBB which is optional) 
try: 
    from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.impl._algorithmic_step_methods import AlgorithmicStepMethods
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.impl._storage_methods import StorageMethods
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.impl._maths_methods import MathsMethods
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.impl._physics_methods import PhysicsMethods
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.impl.condensation_methods import CondensationMethods
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.storage.storage import Storage as ImportedStorage
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.storage.indexed_storage import IndexedStorage as ImportedIndexedStorage
except (NumbaWarning):
    pass
try: 
    from PySDM.backends.numba.random import Random as ImportedRandom
except (NumbaWarning):
    pass

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
