"""
attribute storage class featuring particle permutation logic
"""
from typing import Type

import numpy as np

from PySDM.storages.common.storage import (
    Index,
    Indexed,
    ShapeType,
    Storage,
    StorageSignature,
)


def indexed(storage_cls: Type[Storage]):
    """
    Creates a specialised storage class featuring particle permutation logic.

    Parameters
    ----------
    storage_cls : Type[Storage]
        storage class to be used for the storage

    Returns
    -------
    Type[Indexed]
        specialised storage class featuring particle permutation logic
    """
    assert issubclass(storage_cls, Storage)

    class _IndexedStorage(storage_cls, Indexed):
        def __init__(self, idx: Index, signature: StorageSignature):
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
        def indexed(cls, idx: Index, storage: Storage):
            return cls(
                idx, StorageSignature(storage.data, storage.shape, storage.dtype)
            )

        @classmethod
        def indexed_and_empty(cls, idx: Index, shape: ShapeType, dtype: Type):
            storage = storage_cls.empty(shape, dtype)
            return cls.indexed(idx, storage)

        @classmethod
        def indexed_from_ndarray(cls, idx: Index, array: np.ndarray):
            storage = storage_cls.from_ndarray(array)
            return cls.indexed(idx, storage)

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

    return _IndexedStorage
