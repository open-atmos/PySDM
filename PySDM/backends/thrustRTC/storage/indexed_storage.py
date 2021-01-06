"""
Created at 25.08.2020
"""

import numpy as np

from PySDM.backends.thrustRTC.impl._algorithmic_step_methods import AlgorithmicStepMethods
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
            return result[idx[:len(self)]]

    def read_row(self, i):
        result_data = self.data.range(self.shape[1] * i, self.shape[1] * (i+1))
        result = IndexedStorage(self.idx, result_data, (1, *self.shape[1:]), self.dtype)
        return result
