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
        return len(self.idx)

    def __getitem__(self, item):
        result = Storage.__getitem__(self, item)
        if isinstance(result, Storage):
            return IndexedStorage.indexed(self.idx, result)
        else:
            return result

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

    def to_ndarray(self, *, raw=False):
        result = Storage.to_ndarray(self)
        if raw:
            return result
        else:
            idx = self.idx.to_ndarray()
            return result[idx[:len(self)]]
