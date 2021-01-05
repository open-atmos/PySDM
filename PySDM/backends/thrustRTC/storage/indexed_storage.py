"""
Created at 25.08.2020
"""

import numpy as np

from PySDM.backends.thrustRTC.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.thrustRTC.impl._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.thrustRTC.impl._storage_methods import StorageMethods
from PySDM.backends.thrustRTC.storage.storage import Storage


class IndexedStorage(Storage):

    def __init__(self, idx, data, shape, dtype):
        super().__init__(data, shape, dtype)
        assert idx is not None
        self.idx = idx

    def __len__(self):
        return self.idx.length

    @staticmethod
    def indexed(idx, storage):
        return IndexedStorage(idx, storage.data, storage.shape, storage.dtype)

    @staticmethod
    def empty(idx, shape, dtype):
        storage = Storage.empty(shape, dtype)
        result = IndexedStorage.indexed(idx, storage)
        return result

    @staticmethod
    def from_ndarray(array):
        storage = Storage.from_ndarray(array)
        result = IndexedStorage.indexed(None, storage)
        return result

    def amax(self):
        return AlgorithmicStepMethods.amax(self, self.idx)

    def amin(self):
        return AlgorithmicStepMethods.amin(self, self.idx)

    # def distance_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.distance_pair(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #     self.idx = None
    #
    # def find_pairs(self, cell_start, cell_id):
    #     AlgorithmicStepMethods.find_pairs(cell_start.data, self.data, cell_id.data, cell_id.idx.data, len(cell_id))
    #     self.idx = None
    #     self.length = len(cell_id)
    #
    # def max_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.max_pair(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #     self.idx = None
    #
    # def sort_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.sort_pair(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #     self.idx = None
    #
    # def sum_pair(self, other, is_first_in_pair):
    #     AlgorithmicStepMethods.sum_pair(self.data, other.data, is_first_in_pair.data, other.idx.data, len(other))
    #     self.idx = None

    def to_ndarray(self, *, raw=False):
        self.detach()
        result = self.data.to_host()
        result = np.reshape(result, self.shape)
        if len(result.shape) > 1:
            result = result.squeeze()

        if raw:
            return result
        else:
            idx = self.idx.to_ndarray()
            return result[idx[0:len(self)]]

    def read_row(self, i):
        result_data = self.data.range(self.shape[1] * i, self.shape[1] * (i+1))
        result = IndexedStorage(self.idx, result_data, (1, *self.shape[1:]), self.dtype)
        return result

    def remove_zeros(self):
        self.idx.length = AlgorithmicMethods.remove_zeros(self.data, self.idx.data, self.idx.length)

    def shuffle(self, temporary, parts=None):
        if parts is None:
            StorageMethods.shuffle_global(idx=self.data, length=len(self), u01=temporary.data)
        else:
            StorageMethods.shuffle_local(idx=self.data, u01=temporary.data, cell_start=parts.data)

