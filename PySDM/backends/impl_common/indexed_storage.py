"""
attribute storage class featuring particle permutation logic
"""

from .storage_utils import StorageSignature


def make_IndexedStorage(backend):
    class IndexedStorage(backend.Storage):
        def __init__(self, idx, signature):
            super().__init__(signature)
            assert idx is not None
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, item):
            result = backend.Storage.__getitem__(self, item)
            if isinstance(result, backend.Storage):
                return IndexedStorage.indexed(self.idx, result)
            return result

        @staticmethod
        def indexed(idx, storage):
            return IndexedStorage(
                idx, StorageSignature(storage.data, storage.shape, storage.dtype)
            )

        @staticmethod
        def empty(idx, shape, dtype):
            storage = backend.Storage.empty(shape, dtype)
            result = IndexedStorage.indexed(idx, storage)
            return result

        @staticmethod
        def from_ndarray(idx, array):
            storage = backend.Storage.from_ndarray(array)
            result = IndexedStorage.indexed(idx, storage)
            return result

        def to_ndarray(self, *, raw=False):
            result = backend.Storage.to_ndarray(self)
            dim = len(self.shape)
            if raw:
                return result
            if dim == 1:
                idx = self.idx.to_ndarray()
                return result[idx[: len(self)]]
            if dim == 2:
                idx = self.idx.to_ndarray()
                return result[:, idx[: len(self)]]
            raise NotImplementedError()

    return IndexedStorage
