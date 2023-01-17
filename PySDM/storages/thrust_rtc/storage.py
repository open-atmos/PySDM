"""
storage internals for the ThrustRTC backend
"""
from typing import Any, Callable, Literal, Optional, Type, cast

import numpy as np

from PySDM.storages.common.storage import Storage as BaseStorage
from PySDM.storages.common.storage import StorageSignature
from PySDM.storages.common.utils import get_data_from_ndarray
from PySDM.storages.thrust_rtc.conf import trtc
from PySDM.storages.thrust_rtc.operators import ThrustStorageOperators


class Storage(BaseStorage):
    INT = np.int64
    BOOL = np.bool_
    data: trtc.DVVector

    def __init_subclass__(
        cls,
        conv_function: Callable[[float], Any] = trtc.DVFloat,
        real_type: Literal["double", "float"] = "float",
        np_float_dtype: Type = np.float32,
    ):
        cls._conv_function = lambda _, x: conv_function(x)
        cls._real_type = real_type
        cls._np_dtype = np_float_dtype
        cls.FLOAT = np_float_dtype
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops = ThrustStorageOperators(self._conv_function)

    def __getitem__(self, item):
        raise RuntimeError(
            "Not implemented - the method should be overriden in subclasses"
        )

    def __setitem__(self, key, value):
        if not (
            isinstance(key, slice)
            and key.start is None
            and key.stop is None
            and key.step is None
        ):
            raise NotImplementedError()
        if hasattr(value, "data") and hasattr(value, "shape") and len(value.shape) != 0:
            if isinstance(value, np.ndarray):
                vector = trtc.device_vector_from_numpy(value)
                trtc.Copy(vector, self.data)
            else:
                trtc.Copy(value.data, self.data)
        else:
            if isinstance(value, int):
                dvalue = trtc.DVInt64(value)
            elif isinstance(value, float):
                dvalue = self._conv_function(value)
            else:
                raise TypeError("Only Storage, int and float are supported.")
            trtc.Fill(self.data, dvalue)
        return self

    def __iadd__(self, other):
        self.ops.add(self, other)
        return self

    def __isub__(self, other):
        self.ops.subtract(self, other)
        return self

    def __imul__(self, other):
        self.ops.multiply(self, other)
        return self

    def __itruediv__(self, other):
        self.ops.truediv(self, other)
        return self

    def __imod__(self, other):
        self.ops.row_modulo(self, other)
        return self

    def __ipow__(self, other: int):
        self.ops.power(self, other)
        return self

    def __bool__(self):
        if len(self) == 1:
            result = bool(self.data.to_host()[0] != 0)
        else:
            raise NotImplementedError("Logic value of array is ambiguous.")
        return result

    def _to_host(self):
        if isinstance(self.data, trtc.DVVector.DVRange):
            if self.dtype is self.FLOAT:
                elem_cls = self._real_type
            elif self.dtype is self.INT:
                elem_cls = "int64_t"
            elif self.dtype is self.BOOL:
                elem_cls = "bool"
            else:
                raise NotImplementedError()

            data = trtc.device_vector(elem_cls, self.data.size())

            trtc.Copy(self.data, data)
        else:
            data = self.data
        return data.to_host()

    def amin(self):
        return self.ops.amin(self.data)

    def all(self):
        assert self.dtype is self.BOOL
        return self.amin()

    def download(self, target, reshape=False):
        shape = target.shape if reshape else self.shape
        target[:] = np.reshape(self._to_host(), shape)

    @classmethod
    def _get_empty_data(cls, shape, dtype):
        if dtype in (float, cls.FLOAT):
            elem_cls = cls._real_type
            dtype = cls.FLOAT
        elif dtype in (int, cls.INT):
            elem_cls = "int64_t"
            dtype = cls.INT
        elif dtype in (bool, cls.BOOL):
            elem_cls = "bool"
            dtype = cls.BOOL
        else:
            raise NotImplementedError

        size = int(np.prod(shape))
        data = None if size == 0 else trtc.device_vector(elem_cls, size)
        return StorageSignature(data, shape, dtype)

    @classmethod
    def _get_data_from_ndarray(cls, array: np.ndarray):
        return get_data_from_ndarray(
            array=array,
            storage_class=cls,
            copy_fun=lambda array_astype: trtc.device_vector_from_numpy(
                array_astype.ravel()
            ),
        )

    def floor(self, other: Optional["Storage"] = None):
        if other is None:
            self.ops.floor(self)
        else:
            self.ops.floor_out_of_place(self, other)
        return self

    def product(self, multiplicand: "Storage", multiplier: "Storage"):
        self.ops.multiply_out_of_place(self, multiplicand, multiplier)
        return self

    def ratio(self, dividend: "Storage", divisor: "Storage"):
        self.ops.divide_out_of_place(self, dividend, divisor)
        return self

    def sum(self, arg_a, arg_b):
        self.ops.sum_out_of_place(self, arg_a, arg_b)
        return self

    def ravel(self, other):
        if self.is_storage(other):
            trtc.Copy(other.data, self.data)
        else:
            self.data = trtc.device_vector_from_numpy(other.ravel())

    def to_ndarray(self):
        result = self._to_host()
        result = np.reshape(result, self.shape)
        return result

    def upload(self, data):
        trtc.Copy(
            trtc.device_vector_from_numpy(data.astype(self.dtype).ravel()),
            self.data,
        )

    def divide_if_not_zero(self, divisor):
        self.ops.divide_if_not_zero(self, divisor)
        return self


class ThrustRTCStorageGetItemMixin:

    storage_class: Type[Storage]

    def __getitem__(self, item):
        instance = cast(Storage, self)
        dim = len(instance.shape)
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or instance.shape[0]
            if dim == 1:
                result_data = instance.data.range(start, stop)
                result_shape = (stop - start,)
            elif dim == 2:
                result_data = instance.data.range(
                    instance.shape[1] * start, instance.shape[1] * stop
                )
                result_shape = (stop - start, instance.shape[1])
            else:
                raise NotImplementedError(
                    "Only 2 or less dimensions array is supported."
                )
            return self.storage_class(
                StorageSignature(result_data, result_shape, instance.dtype)
            )
        elif (
            dim == 2
            and isinstance(item, tuple)
            and isinstance(item[0], int)
            and isinstance(item[1], slice)
        ):
            assert item[1].start is None or item[1].start == 0
            assert item[1].stop is None or item[1].stop == instance.shape[1]
            assert item[1].step is None or item[1].step == 1
            result_data = instance.data.range(
                instance.shape[1] * item[0], instance.shape[1] * (item[0] + 1)
            )
            return self.storage_class(
                StorageSignature(result_data, (*instance.shape[1:],), instance.dtype)
            )
        else:
            return instance.to_ndarray()[item]


class BaseFloatStorage(Storage):
    pass


class FloatStorage(ThrustRTCStorageGetItemMixin, BaseFloatStorage):
    storage_class = BaseFloatStorage


class BaseDoubleStorage(
    Storage, conv_function=trtc.DVDouble, real_type="double", np_float_dtype=np.float64
):
    pass


class DoubleStorage(ThrustRTCStorageGetItemMixin, BaseDoubleStorage):
    storage_class = BaseDoubleStorage
