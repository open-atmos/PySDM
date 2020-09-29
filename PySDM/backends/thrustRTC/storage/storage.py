"""
Created at 30.05.2020
"""

import numpy as np
from PySDM.backends.thrustRTC.storage import storage_impl as impl
from ..conf import trtc


class Storage:

    FLOAT = np.float64
    INT = np.int64

    def __init__(self, data, shape, dtype):
        self.data = data
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype

    def __getitem__(self, item):
        if isinstance(item, slice):
            dim = len(self.shape)
            start = item.start or 0
            stop = item.stop or self.shape[0]
            if dim == 1:
                result_data = self.data.range(start, stop)
                result_shape = (stop - start,)
            elif dim == 2:
                result_data = self.data.range(self.shape[1] * start, self.shape[1] * stop)
                result_shape = (stop - start, self.shape[1])
            else:
                raise NotImplementedError("Only 2 or less dimensions array is supported.")
            result = Storage(result_data, result_shape, self.dtype)
        else:
            result = self.to_ndarray()[item]
        return result

    def __setitem__(self, key, value):
        if hasattr(value, 'data'):
            trtc.Copy(value.data, self.data)
        else:
            if isinstance(value, int):
                dvalue = trtc.DVInt64(value)
            elif isinstance(value, float):
                dvalue = trtc.DVDouble(value)
            else:
                raise TypeError("Only Storage, int and float are supported.")
            trtc.Fill(self.data, dvalue)
        return self

    def __add__(self, other):
        raise NotImplementedError("Use +=")

    def __iadd__(self, other):
        impl.add(self, other)
        return self

    def __sub__(self, other):
        raise NotImplementedError("Use -=")

    def __isub__(self, other):
        impl.subtract(self, other)
        return self

    def __mul__(self, other):
        raise NotImplementedError("Use *=")

    def __imul__(self, other):
        impl.multiply(self, other)
        return self

    def __mod__(self, other):
        raise NotImplementedError("Use %=")

    def __imod__(self, other):
        # TODO
        impl.row_modulo(self, other)
        return self

    def __pow__(self, other):
        raise NotImplementedError("Use **=")

    def __ipow__(self, other):
        impl.power(self, other)
        return self

    def __len__(self):
        return self.shape[0]

    def __bool__(self):
        if len(self) == 1:
            result = bool(self.data.to_host()[0] != 0)
        else:
            raise NotImplementedError("Logic value of array is ambiguous.")
        return result

    def detach(self):
        if isinstance(self.data, trtc.DVVector.DVRange):
            if self.dtype is Storage.FLOAT:
                elem_cls = 'double'
            elif self.dtype is Storage.INT:
                elem_cls = 'int64_t'
            else:
                raise NotImplementedError()

            data = trtc.device_vector(elem_cls, self.data.size())

            trtc.Copy(self.data, data)
            self.data = data

    def download(self, target, reshape=False):
        shape = target.shape if reshape else self.shape
        self.detach()
        target[:] = np.reshape(self.data.to_host(), shape)

    @staticmethod
    def empty(shape, dtype):
        if dtype in (float, Storage.FLOAT):
            elem_cls = 'double'
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT):
            elem_cls = 'int64_t'
            dtype = Storage.INT
        else:
            raise NotImplementedError

        data = trtc.device_vector(elem_cls, int(np.prod(shape)))
        result = Storage(data, shape, dtype)
        return result

    @staticmethod
    def from_ndarray(array):
        if str(array.dtype).startswith('int'):
            dtype = Storage.INT
        elif str(array.dtype).startswith('float'):
            dtype = Storage.FLOAT
        else:
            raise NotImplementedError()

        data = trtc.device_vector_from_numpy(array.astype(dtype).ravel())
        result = Storage(data, array.shape, dtype)
        return result

    def floor(self, other=None):
        if other is None:
            impl.floor(self.data)
        else:
            impl.floor_out_of_place(self, other)
        return self

    def product(self, multiplicand, multiplier):
        impl.multiply_out_of_place(self, multiplicand, multiplier)
        return self

    def ravel(self, other):
        if isinstance(other, Storage):
            trtc.Copy(other.data, self.data)
        else:
            self.data = trtc.device_vector_from_numpy(other.ravel())

    # TODO: handle by getitem
    def read_row(self, i):
        start = self.shape[1] * i
        stop = start + self.shape[1]
        result_data = self.data.range(start, stop)
        result = Storage(result_data, self.shape[1:], self.dtype)
        return result

    def to_ndarray(self):
        self.detach()
        result = self.data.to_host()
        result = np.reshape(result, self.shape)
        return result

    def urand(self, generator=None):
        generator(self)

    def upload(self, data):
        trtc.Copy(trtc.device_vector_from_numpy(data.ravel()), self.data)

    def write_row(self, i, row):
        start = self.shape[1] * i
        stop = start + self.shape[1]
        trtc.Copy(row.data, self.data.range(start, stop))
