"""
Created at 09.11.2020
"""

import numpy as np


def make_Index(backend):

    class Index(backend.Storage):

        def __init__(self, data, length):
            assert isinstance(length, int)
            super().__init__(data, length, backend.Storage.INT)
            self.length = backend.Storage.INT(length)

        def __len__(self):
            return self.length

        @staticmethod
        def identity_index(length):
            result = Index.from_ndarray(np.arange(length, dtype=backend.Storage.INT))
            return result

        def reset_index(self):
            backend.identity_index(self.data)

        @staticmethod
        def empty(*args, **kwargs):
            raise TypeError("'Index' class cannot be instantiated as empty.")

        @staticmethod
        def from_ndarray(array):
            data, array.shape, _ = backend.Storage._get_data_from_ndarray(array)
            result = Index(data, array.shape[0])
            return result

        def sort_by_key(self, keys):
            backend.sort_by_key(self, keys)

        def shuffle(self, temporary, parts=None):
            if parts is None:
                backend.shuffle_global(idx=self.data, length=self.length, u01=temporary.data)
            else:
                backend.shuffle_local(idx=self.data, u01=temporary.data, cell_start=parts.data)

        def remove_zeros(self, indexed_storage):
            self.length = backend.remove_zeros(indexed_storage.data, self.data, self.length)

    return Index
