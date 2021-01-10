"""
Created at 03.06.2020
"""

from PySDM.backends.numba.impl._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.numba.storage.storage import Storage


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
    def from_ndarray(idx, array):
        storage = Storage.from_ndarray(array)
        result = IndexedStorage.indexed(idx, storage)
        return result

    def amax(self):
        return AlgorithmicStepMethods.amax(self.data, self.idx.data, len(self))

    def amin(self):
        return AlgorithmicStepMethods.amin(self.data, self.idx.data, len(self))

    def to_ndarray(self, *, raw=False):
        if raw:
            return self.data.copy()
        else:
            return self.data[self.idx.data[:len(self)]].copy()

    def read_row(self, i):
        # TODO #342 shape like in ThrustRTC
        result = IndexedStorage(self.idx, self.data[i, :], *self.shape[1:], self.dtype)
        return result
