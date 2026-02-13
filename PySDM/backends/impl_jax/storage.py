"""
CPU Numpy-based implementation of Storage class
"""

import jax.numpy as jnp
import numpy as np
import jax
import time

from PySDM.backends.impl_common.storage_utils import (
    StorageBase,
    StorageSignature,
    empty,
    get_data_from_ndarray,
)
from PySDM.backends.impl_jax import storage_impl as impl


class Storage(StorageBase):
    FLOAT = jnp.float64
    INT = jnp.int64
    BOOL = jnp.bool_

    
    def ravel(self, other):
        print("Begin ravel")
        print(self.data)
        if isinstance(other, Storage):
            self.data = other.data.ravel()
        else:
            self.data = other.ravel()
        print(self.data)
        print("End ravel")

    def at(self, index):
        assert self.shape == (1,), "Cannot call at() on Storage of shape other than (1,)"
        return self.data[index]

    def __imul__(self, other):
        if hasattr(other, "data"):
            impl.multiply(self.data, other.data)
        else:
            impl.multiply(self.data, other)
        return self

    def __itruediv__(self, other):
        if hasattr(other, "data"):
            self.data[:] /= other.data[:]
        else:
            self.data[:] /= other
        return self

    def download(self, target, reshape=False):
        if reshape:
            data = self.data.reshape(target.shape)
        else:
            data = self.data
        # target = jnp.asarray(data)
        # Not sure here?
        np.copyto(target, np.asarray(data), casting="safe")

    @staticmethod
    def _get_empty_data(shape, dtype):
        if dtype in (float, Storage.FLOAT):
            data = jnp.full(shape, jnp.nan, dtype=Storage.FLOAT)
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT):
            data = jnp.full(shape, -1, dtype=Storage.INT)
            dtype = Storage.INT
        elif dtype in (bool, Storage.BOOL):
            data = jnp.full(shape, -1, dtype=Storage.BOOL)
            dtype = Storage.BOOL
        else:
            raise NotImplementedError()

        return StorageSignature(data, shape, dtype)

    @staticmethod
    def empty(shape, dtype):
        return empty(shape, dtype, Storage)

    @staticmethod
    def _get_data_from_ndarray(array):
        
        return get_data_from_ndarray(
            array=array,
            storage_class=Storage,
            copy_fun=lambda array_astype: array_astype.copy(),
        )

    @staticmethod
    def from_ndarray(array):
        result = Storage(Storage._get_data_from_ndarray(array))
        return result

    def urand(self, generator):
        generator(self)


    def upload(self, data):
        self.fill(data)

    def fill(self, other):
        if isinstance(other, Storage):
            # self.data[:] = other.data
            # self.data.at[:].set(other.data)
            self.data = other.data
        else:
            # self.data[:] = other
            # self.data.at[:].set(other)
            self.data = other

    def to_ndarray(self):
        return np.array(self.data)
