"""
permutation-defining Index class (can be shared between multiple IndexedStorage instances)
"""
from typing import Type, TypeVar

import numpy as np

from PySDM.storages.common.backend import IndexBackend
from PySDM.storages.common.storage import Index, Storage, StorageSignature

BackendType = TypeVar("BackendType", bound=IndexBackend)


def index(backend: BackendType, storage_cls: Type[Storage]):
    assert issubclass(storage_cls, Storage)

    class _Index(storage_cls, Index):
        def __init__(self, data: np.ndarray, length: int):
            assert isinstance(length, int)
            self.length = storage_cls.INT(length)
            super().__init__(StorageSignature(data, length, storage_cls.INT))

        def __len__(self):
            return self.length

        @classmethod
        def identity_index(cls, length: int):
            return cls.from_ndarray(np.arange(length, dtype=cls.INT))

        def reset_index(self):
            backend.identity_index(self.data)

        @staticmethod
        def empty(*args, **kwargs):
            raise TypeError("'Index' class cannot be instantiated as empty.")

        @classmethod
        def from_ndarray(cls, array: np.ndarray) -> "_Index":
            data, array.shape, _ = cls._get_data_from_ndarray(array)
            return cls(data, array.shape[0])

        def sort_by_key(self, keys: Storage) -> None:
            backend.sort_by_key(self, keys)

        def shuffle(self, temporary, parts=None):
            if parts is None:
                backend.shuffle_global(
                    idx=self.data, length=self.length, u01=temporary.data
                )
            else:
                backend.shuffle_local(
                    idx=self.data, u01=temporary.data, cell_start=parts.data
                )

        def remove_zero_n_or_flagged(self, indexed_storage: Storage):
            self.length = backend.remove_zero_n_or_flagged(
                indexed_storage.data, self.data, self.length
            )

    return _Index
