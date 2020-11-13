"""
Created at 09.11.2020
"""

from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.numba.impl._storage_methods import StorageMethods
from PySDM.backends.numba.storage.storage import Storage


class Index(Storage):

    def __init__(self, data, shape, dtype):
        assert len(shape) == 1
        super().__init__(data, shape, dtype)
        self.length = shape[0]

    def __len__(self):
        return self.length

    @staticmethod
    def empty(shape, dtype):
        result = Index(*Storage._get_empty_data(shape, dtype))
        return result

    @staticmethod
    def from_ndarray(array):
        result = Index(*Storage._get_data_from_ndarray(array))
        return result

    def shuffle(self, temporary, parts=None):
        if parts is None:
            StorageMethods.shuffle_global(idx=self.data, length=self.length, u01=temporary.data)
        else:
            StorageMethods.shuffle_local(idx=self.data, u01=temporary.data, cell_start=parts.data)

    def remove_zeros(self, indexed_storage):
        self.length = AlgorithmicMethods.remove_zeros(indexed_storage.data, self.data, self.length)
