"""
Created at 03.06.2020
"""

from .storage import Storage
from ._storage_methods import StorageMethods
from ._algorithmic_step_methods import AlgorithmicStepMethods
from ._algorithmic_methods import AlgorithmicMethods


class IndexedStorage(Storage):

    def __init__(self, idx, data, shape, dtype):
        super().__init__(data, shape, dtype)
        self.idx = idx
        self.length = self.shape[0] if self.idx is None else self.idx.length

    def __len__(self):
        return self.length if self.idx is None else self.idx.length

    @staticmethod
    def indexed(idx, storage):
        return IndexedStorage(idx, storage.data, storage.shape, storage.dtype)

    @staticmethod
    def empty(shape, dtype):
        storage = Storage.empty(shape, dtype)
        result = IndexedStorage.indexed(None, storage)
        return result

    @staticmethod
    def from_ndarray(array):
        storage = Storage.from_ndarray(array)
        result = IndexedStorage.indexed(None, storage)
        return result

    def amax(self):
        return AlgorithmicStepMethods.amax(self.data, self.idx.data, self.length)

    def amin(self):
        return AlgorithmicStepMethods.amin(self.data, self.idx.data, self.length)

    def distance_pair(self, other, is_first_in_pair):
        AlgorithmicStepMethods.distance_pair(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
        self.idx = None

    def find_pairs(self, cell_start, cell_id):
        AlgorithmicStepMethods.find_pairs_body(cell_start.data, self.data, cell_id.data, cell_id.idx.data, len(cell_id))
        self.idx = None
        self.length = len(cell_id)

    def max_pair(self, other, is_first_in_pair):
        AlgorithmicStepMethods.max_pair_body(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
        self.idx = None

    def sum_pair(self, other, is_first_in_pair):
        AlgorithmicStepMethods.sum_pair_body(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
        self.idx = None

    def read_row(self, i):
        result = IndexedStorage(self.idx, self.data[i, :], *self.shape[1:], self.dtype)
        return result

    def remove_zeros(self):
        self.idx.length = AlgorithmicMethods.remove_zeros(self.data, self.idx.data, self.length)
        self.idx = None

    def shuffle(self, temporary, parts=None):
        if parts is None:
            StorageMethods.shuffle_global(idx=self.data, length=self.length, u01=temporary.data)
        else:
            StorageMethods.shuffle_local(idx=self.data, u01=temporary.data, cell_start=parts.data)
