"""
CPU Numpy-based implementation of Storage class
"""

import numpy as np

from PySDM.backends.impl_common.storage_utils import (
    StorageBase,
    StorageSignature,
    empty,
    get_data_from_ndarray,
)
from PySDM.backends.impl_numba import storage_impl as impl


class Storage(StorageBase):
    FLOAT = np.float64
    INT = np.int64
    BOOL = np.bool_

    def __getitem__(self, item):
        dim = len(self.shape)
        if isinstance(item, slice):
            step = item.step or 1
            if step != 1:
                raise NotImplementedError("step != 1")
            start = item.start or 0
            if dim == 1:
                stop = item.stop or len(self)
                result_data = self.data[item]
                result_shape = (stop - start,)
            elif dim == 2:
                stop = item.stop or self.shape[0]
                result_data = self.data[item]
                result_shape = (stop - start, self.shape[1])
            else:
                raise NotImplementedError(
                    "Only 2 or less dimensions array is supported."
                )
            if stop > self.data.shape[0]:
                raise IndexError(
                    f"requested a slice ({start}:{stop}) of Storage"
                    f" with first dim of length {self.data.shape[0]}"
                )
            result = Storage(StorageSignature(result_data, result_shape, self.dtype))
        elif isinstance(item, tuple) and dim == 2 and isinstance(item[1], slice):
            result = Storage(
                StorageSignature(self.data[item[0]], (*self.shape[1:],), self.dtype)
            )
        else:
            result = self.data[item]
        return result

    def __setitem__(self, key, value):
        if hasattr(value, "data"):
            self.data[key] = value.data
        else:
            self.data[key] = value
        return self

    def __iadd__(self, other):
        if isinstance(other, Storage):
            impl.add(self.data, other.data)
        elif (
            isinstance(other, tuple)
            and len(other) == 3
            and isinstance(other[0], float)
            and other[1] == "*"
            and isinstance(other[2], Storage)
        ):
            impl.add_with_multiplier(self.data, other[2].data, other[0])
        else:
            impl.add(self.data, other)
        return self

    def __isub__(self, other):
        impl.subtract(self.data, other.data)
        return self

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

    def __imod__(self, other):
        impl.row_modulo(self.data, other.data)
        return self

    def __ipow__(self, other):
        impl.power(self.data, other)
        return self

    def __bool__(self):
        if len(self) == 1:
            result = bool(self.data[0] != 0)
        else:
            raise NotImplementedError("Logic value of array is ambiguous.")
        return result

    def detach(self):
        if self.data.base is not None:
            self.data = np.array(self.data)

    def download(self, target, reshape=False):
        if reshape:
            data = self.data.reshape(target.shape)
        else:
            data = self.data
        np.copyto(target, data, casting="safe")

    @staticmethod
    def _get_empty_data(shape, dtype):
        if dtype in (float, Storage.FLOAT):
            data = np.full(shape, np.nan, dtype=Storage.FLOAT)
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT):
            data = np.full(shape, -1, dtype=Storage.INT)
            dtype = Storage.INT
        elif dtype in (bool, Storage.BOOL):
            data = np.full(shape, -1, dtype=Storage.BOOL)
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

    def amin(self):
        return impl.amin(self.data)

    def amax(self):
        return impl.amax(self.data)

    def all(self):
        return self.data.all()

    @staticmethod
    def from_ndarray(array):
        result = Storage(Storage._get_data_from_ndarray(array))
        return result

    def floor(self, other=None):
        if other is None:
            impl.floor(self.data)
        else:
            impl.floor_out_of_place(self.data, other.data)
        return self

    def product(self, multiplicand, multiplier):
        if hasattr(multiplier, "data"):
            impl.multiply_out_of_place(self.data, multiplicand.data, multiplier.data)
        else:
            impl.multiply_out_of_place(self.data, multiplicand.data, multiplier)
        return self

    def ratio(self, dividend, divisor):
        impl.divide_out_of_place(self.data, dividend.data, divisor.data)
        return self

    def divide_if_not_zero(self, divisor):
        impl.divide_if_not_zero(self.data, divisor.data)
        return self

    def sum(self, arg_a, arg_b):
        impl.sum_out_of_place(self.data, arg_a.data, arg_b.data)
        return self

    def ravel(self, other):
        if isinstance(other, Storage):
            self.data[:] = other.data.ravel()
        else:
            self.data[:] = other.ravel()

    def urand(self, generator):
        generator(self)

    def to_ndarray(self):
        return self.data.copy()

    def upload(self, data):
        np.copyto(self.data, data, casting="safe")

    def fill(self, other):
        if isinstance(other, Storage):
            self.data[:] = other.data
        else:
            self.data[:] = other

    def exp(self):
        self.data[:] = np.exp(self.data)
