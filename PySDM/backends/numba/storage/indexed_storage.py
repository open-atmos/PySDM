"""
Created at 03.06.2020
"""

from PySDM.backends.numba.impl._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.numba.storage.storage import Storage


class IndexedStorage(Storage):

    def __init__(self, idx, data, shape, dtype):
        super().__init__(data, shape, dtype)
        self.idx = idx

    def __len__(self):
        return self.idx.length

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
        return AlgorithmicStepMethods.amax(self.data, self.idx.data, len(self))

    def amin(self):
        return AlgorithmicStepMethods.amin(self.data, self.idx.data, len(self))

    # def distance_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.distance_pair(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #     self.idx = None
    #
    # def find_pairs(self, cell_start, cell_id):
    #     AlgorithmicStepMethods.find_pairs_body(cell_start.data, self.data, cell_id.data, cell_id.idx.data, len(cell_id))
    #     self.idx = None
    #     self.length = len(cell_id)
    #
    # def max_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.max_pair_body(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #     self.idx = None
    #
    # def sort_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.sort_pair_body(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #
    # def sum_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.sum_pair_body(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #     self.idx = None

    def to_ndarray(self):
        return self.data[:len(self)].copy()

    def read_row(self, i):
        # TODO: shape like in ThrustRTC
        result = IndexedStorage(self.idx, self.data[i, :], *self.shape[1:], self.dtype)
        return result
