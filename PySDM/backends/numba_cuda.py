"""
Multi-threaded GPU backend using LLVM-powered just-in-time compilation
"""

# from PySDM.backends.impl_numba.random import Random as ImportedRandom
from PySDM.backends.impl_numba_cuda.storage import Storage as ImportedStorage
from PySDM.formulae import Formulae


class NumbaCUDA:
    Storage = ImportedStorage
    # Random = ImportedRandom

    default_croupier = "local"

    def __init__(self, formulae=None, double_precision=True):
        if not double_precision:
            raise NotImplementedError()
        self.formulae = formulae or Formulae()

