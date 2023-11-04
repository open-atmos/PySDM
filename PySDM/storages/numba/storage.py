"""
CPU Numpy-based implementation of Storage class
"""
from numbers import Number
from typing import Optional, Union

import numpy as np

from PySDM.storages.common.storage import Storage as BaseStorage
from PySDM.storages.common.storage import StorageSignature
from PySDM.storages.common.utils import get_data_from_ndarray
from PySDM.storages.numba import operators as ops


class Storage(BaseStorage):

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
            return Storage(StorageSignature(result_data, result_shape, self.dtype))
        elif isinstance(item, tuple) and dim == 2 and isinstance(item[1], slice):
            return Storage(
                StorageSignature(self.data[item[0]], (*self.shape[1:],), self.dtype)
            )

        else:
            return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value.data if self.is_storage(value) else value
        return self

    def __iadd__(self, other: Union["Storage", np.ndarray, Number]) -> "Storage":
        if isinstance(other, Storage):
            ops.add(self.data, other.data)
        else:
            ops.add(self.data, other)
        return self

    def __isub__(self, other: "Storage") -> "Storage":
        ops.subtract(self.data, other.data)
        return self

    def __imul__(self, other: Union["Storage", np.ndarray]) -> "Storage":
        if self.is_storage(other):
            ops.multiply(self.data, other.data)
        else:
            ops.multiply(self.data, other)
        return self

    def __itruediv__(self, other: Union["Storage", np.ndarray]):
        if self.is_storage(other):
            self.data[:] /= other.data[:]
        else:
            self.data[:] /= other
        return self

    def __imod__(self, other: "Storage"):
        ops.row_modulo(self.data, other.data)
        return self

    def __ipow__(self, other: np.ndarray):
        ops.power(self.data, other)
        return self

    def __bool__(self):
        if len(self) == 1:
            result = self.data[0] != 0
        else:
            raise NotImplementedError("Logic value of array is ambiguous.")
        return bool(result)

    def detach(self):
        if self.data.base is not None:
            self.data = np.array(self.data)

    def download(self, target, reshape=False):
        data = self.data.reshape(target.shape) if reshape else self.data
        np.copyto(target, data, casting="safe")

    @classmethod
    def _get_empty_data(cls, shape, dtype):
        if dtype in (float, cls.FLOAT):
            data = np.full(shape, -1.0, dtype=cls.FLOAT)
            dtype = cls.FLOAT
        elif dtype in (int, cls.INT):
            data = np.full(shape, -1, dtype=cls.INT)
            dtype = cls.INT
        elif dtype in (bool, cls.BOOL):
            data = np.full(shape, -1, dtype=cls.BOOL)
            dtype = cls.BOOL
        else:
            raise NotImplementedError()

        return StorageSignature(data, shape, dtype)

    @classmethod
    def _get_data_from_ndarray(cls, array):
        return get_data_from_ndarray(
            array=array,
            storage_class=cls,
            copy_fun=lambda array_astype: array_astype.copy(),
        )

    def amin(self) -> Union[FLOAT, INT, BOOL]:
        return ops.amin(self.data)

    def all(self) -> bool:
        return self.data.all()

    def floor(self, other: Optional["Storage"] = None):
        if other is None:
            ops.floor(self.data)
        else:
            ops.floor_out_of_place(self.data, other.data)
        return self

    def product(self, multiplicand, multiplier):
        if self.is_storage(multiplier):
            ops.multiply_out_of_place(self.data, multiplicand.data, multiplier.data)
        else:
            ops.multiply_out_of_place(self.data, multiplicand.data, multiplier)
        return self

    def ratio(self, dividend, divisor):
        ops.divide_out_of_place(self.data, dividend.data, divisor.data)
        return self

    def divide_if_not_zero(self, divisor):
        ops.divide_if_not_zero(self.data, divisor.data)
        return self

    def sum(self, arg_a, arg_b):
        ops.sum_out_of_place(self.data, arg_a.data, arg_b.data)
        return self

    def ravel(self, other):
        self.data[:] = other.data.ravel() if self.is_storage(other) else other.ravel()

    def to_ndarray(self):
        return self.data.copy()

    def upload(self, data):
        np.copyto(self.data, data, casting="safe")
