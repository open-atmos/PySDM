"""
attribute storage class featuring particle permutation logic
"""
from typing import Type, cast

import numpy as np

from PySDM.storages.common.storage import (
    Indexed,
    IndexedStorage,
    Storage,
    StorageSignature,
)


def indexed(storage_cls: Type[Storage]) -> Type[IndexedStorage]:
    assert issubclass(storage_cls, Storage)

    class _IndexedStorage(storage_cls, Indexed):
        def __init__(self, idx, signature: StorageSignature):
            super().__init__(signature)
            assert idx is not None
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, item):
            result = super().__getitem__(item)
            if isinstance(result, Storage):
                return self.indexed(self.idx, result)
            return result

        @classmethod
        def indexed(cls, idx, storage: Storage):
            return cls(
                idx, StorageSignature(storage.data, storage.shape, storage.dtype)
            )

        @classmethod
        def indexed_and_empty(cls, idx, shape, dtype):
            storage = cls.empty(shape, dtype)
            result = cls.indexed(idx, storage)
            return result

        @classmethod
        def indexed_from_ndarray(cls, idx, array):
            storage = cls.from_ndarray(array)
            result = cls.indexed(idx, storage)
            return result

        def to_ndarray(self, *, raw=False) -> np.ndarray:
            result = super().to_ndarray()
            if raw:
                return result
            dim = len(self.shape)
            if dim == 1:
                idx = self.idx.to_ndarray()
                return result[idx[: len(self)]]
            if dim == 2:
                idx = self.idx.to_ndarray()
                return result[:, idx[: len(self)]]
            raise NotImplementedError()

    return cast(type(IndexedStorage), _IndexedStorage)
