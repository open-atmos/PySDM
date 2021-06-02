def make_IndexedStorage(backend):
    class IndexedStorage(backend.Storage):

        def __init__(self, idx, data, shape, dtype):
            super().__init__(data, shape, dtype)
            assert idx is not None
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, item):
            result = backend.Storage.__getitem__(self, item)
            if isinstance(result, backend.Storage):
                return IndexedStorage.indexed(self.idx, result)
            else:
                return result

        @staticmethod
        def indexed(idx, storage):
            return IndexedStorage(idx, storage.data, storage.shape, storage.dtype)

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
            elif dim == 1:
                idx = self.idx.to_ndarray()
                return result[idx[:len(self)]]
            elif dim == 2:
                idx = self.idx.to_ndarray()
                return result[:, idx[:len(self)]]
            else:
                raise NotImplementedError()

    return IndexedStorage
