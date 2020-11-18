"""
Created at 09.11.2020
"""

import numpy as np

from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.numba.impl._storage_methods import StorageMethods
from PySDM.backends.numba.storage.storage import Storage


class Index(Storage):

    def __init__(self, data, length):
        assert isinstance(length, int)
        super().__init__(data, length, int)
        self.length = length

    def __len__(self):
        return self.length

    @staticmethod
    def empty(length):
        result = Index(np.arange(length, dtype=Storage.INT), length)
        return result

    @staticmethod
    def from_ndarray(array):
        data, array.shape, _ = Storage._get_data_from_ndarray(array)
        result = Index(data, array.shape[0])
        return result

    def shuffle(self, temporary, parts=None):
        if parts is None:
            StorageMethods.shuffle_global(idx=self.data, length=self.length, u01=temporary.data)
        else:
            StorageMethods.shuffle_local(idx=self.data, u01=temporary.data, cell_start=parts.data)

    def remove_zeros(self, indexed_storage):
        self.length = AlgorithmicMethods.remove_zeros(indexed_storage.data, self.data, self.length)
