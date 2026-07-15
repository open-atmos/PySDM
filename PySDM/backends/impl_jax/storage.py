"""
JAX-based implementation of Storage class
"""

import jax.numpy as jnp
import numpy as np

from PySDM.backends.impl_common.storage_utils import (
    StorageBase,
    StorageSignature,
    empty,
    get_data_from_ndarray,
)
from PySDM.backends.impl_jax import storage_impl as impl


class Storage(StorageBase):
    # JAX and Numpy types here so that Storage dtypes identify as numpy dtypes
    FLOAT = np.float64
    JAX_FLOAT = jnp.float64

    INT = np.int64
    JAX_INT = jnp.int64

    BOOL = np.bool_
    JAX_BOOL = jnp.bool_

    def ravel(self, other):
        if isinstance(other, Storage):
            self.data = other.data.ravel()
        else:
            self.data = other.ravel()

    def at(self, index):
        assert self.shape == (
            1,
        ), "Cannot call at() on Storage of shape other than (1,)"
        return self.data[index]

    def __imul__(self, other):
        if hasattr(other, "data"):
            self.data = impl.multiply(self.data, other.data).block_until_ready()
        else:
            self.data = impl.multiply(self.data, other).block_until_ready()
        return self

    def __itruediv__(self, other):
        if hasattr(other, "data"):
            self.data = jnp.true_divide(self.data, other.data)
        else:
            self.data = jnp.true_divide(self.data, other)
        return self

    def download(self, target, reshape=False):
        if reshape:
            data = self.data.reshape(target.shape)
        else:
            data = self.data
        np.copyto(target, np.asarray(data), casting="safe")

    @staticmethod
    def _get_empty_data(shape, dtype):
        if dtype in (float, Storage.FLOAT, Storage.JAX_FLOAT):
            data = jnp.full(shape, jnp.nan, dtype=Storage.JAX_FLOAT)
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT, Storage.JAX_INT):
            data = jnp.full(shape, -1, dtype=Storage.JAX_INT)
            dtype = Storage.INT
        elif dtype in (bool, Storage.BOOL, Storage.JAX_BOOL):
            data = jnp.full(shape, -1, dtype=Storage.JAX_BOOL)
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
            copy_fun=lambda array_astype: jnp.array(
                array_astype
            ),  # pylint: disable=unnecessary-lambda
        )

    @staticmethod
    def from_ndarray(array):
        result = Storage(Storage._get_data_from_ndarray(array))
        return result

    def upload(self, data):
        self.fill(data)

    def fill(self, other):
        if isinstance(other, Storage):
            self.data = self.data.at[:].set(other.data)
        else:
            self.data = self.data.at[:].set(other)

    def row_view(self, i):
        return RowStorage(
            StorageSignature(self.data, (*self.shape[1:],), self.dtype), i, self
        )

    def to_ndarray(self):
        return np.array(self.data)

    def ratio(self, dividend, divisor):
        self.data = impl.divide_out_of_place(
            self.data, dividend.data, divisor.data
        ).block_until_ready()


class RowStorage(Storage):
    """RowStorage is a Storage class that keeps its data attribute as a reference to another
    storage, as well as a stored index to identify which row it is pointing to.

    This was done because JAX does not allow a compatible API for having a reference to a part
    of a storage.
    """

    def __init__(self, signature, row, parent):
        self.parent = parent
        self.row = row
        super().__init__(StorageSignature(self.data, signature.shape, signature.dtype))

    @property
    def data(self):
        return self.parent.data[self.row]

    @data.setter
    def data(self, value):
        self.parent.data = self.parent.data.at[self.row].set(value)
