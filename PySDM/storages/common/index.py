"""
permutation-defining Index class (can be shared between multiple IndexedStorage instances)
"""
from typing import Type, TypeVar, Union

import numpy as np

from PySDM.storages.common.backend import IndexBackend
from PySDM.storages.common.storage import Index, Storage, StorageSignature
from PySDM.storages.thrust_rtc.conf import trtc

BackendType = TypeVar("BackendType", bound=IndexBackend)
DataType = Union[np.ndarray, trtc.DVVector]


def index(backend: BackendType, storage_cls: Type[Storage]):
    """
    Creates a specialised index class featuring particle indexing logic.

    Parameters
    ----------
    backend : BackendType
        backend to be used for the indexing operations
    storage_cls : Type[Storage]
        storage class to be used for the storage
    Returns
    -------
    Type[Index]
        specialised index class featuring particle indexing logic.
    """
    assert issubclass(storage_cls, Storage)

    class _Index(storage_cls, Index):
        def __init__(self, data: DataType, length: int):
            """
            Parameters
            ----------
            data : DataType
                data to be used for the index
            length : int
                number of particles in the index
            """
            assert isinstance(length, int)
            self.length = storage_cls.INT(length)
            super().__init__(StorageSignature(data, length, storage_cls.INT))

        def __len__(self):
            return self.length

        @classmethod
        def identity_index(cls, length: int):
            """
            Parameters
            ----------
            length : int
                number of particles in the index

            Returns
            -------
            _Index
                identity index
            """
            return cls.from_ndarray(np.arange(length, dtype=cls.INT))

        def reset_index(self):
            """
            Resets the index to the identity index.

            Returns
            -------
            None
            """
            backend.identity_index(self.data)

        @staticmethod
        def empty(*args, **kwargs):
            """
            Raises
            ------
            TypeError
                always
            """
            raise TypeError("'Index' class cannot be instantiated as empty.")

        @classmethod
        def from_ndarray(cls, array: np.ndarray) -> "_Index":
            """
            Parameters
            ----------
            array : np.ndarray
                array to be used for the index

            Returns
            -------
            _Index
                index initialised with the given array
            """
            data, array.shape, _ = cls._get_data_from_ndarray(array)
            return cls(data, array.shape[0])

        def sort_by_key(self, keys: Storage) -> None:
            """
            Sorts the index by the given keys.

            Parameters
            ----------
            keys : Storage
                keys to be used for sorting

            Returns
            -------
            None
            """
            backend.sort_by_key(self, keys)

        def shuffle(self, temporary, parts=None):
            """
            Shuffles the index.

            Parameters
            ----------
            temporary : Storage
                temporary storage to be used for the shuffling
            parts : Storage
                parts to be shuffled

            Returns
            -------
            None
            """
            if parts is None:
                backend.shuffle_global(
                    idx=self.data, length=self.length, u01=temporary.data
                )
            else:
                backend.shuffle_local(
                    idx=self.data, u01=temporary.data, cell_start=parts.data
                )

        def remove_zero_n_or_flagged(self, indexed_storage: Storage):
            """
            Removes particles with zero n or flagged.

            Parameters
            ----------
            indexed_storage : Storage
                an indexed storage to be used for the removal

            Returns
            -------
            None
            """
            self.length = backend.remove_zero_n_or_flagged(
                indexed_storage.data, self.data, self.length
            )

    return _Index
