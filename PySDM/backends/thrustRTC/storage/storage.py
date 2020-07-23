"""
Created at 30.05.2020
"""

import numpy as np
from PySDM.backends.thrustRTC.storage import storage_impl as impl
import ThrustRTC as trtc


class Storage:

    FLOAT = np.float64
    INT = np.int64

    def __init__(self, data, shape, dtype):
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, item):
        if isinstance(item, slice):
            dim = len(self.shape)
            if dim == 1:
                result_data = self.data.range(item.start, item.stop)
                result_shape = (item.stop - item.start,)
            elif dim == 2:
                result_data = self.data.range(self.shape[1] * item.start, self.shape[1] * item.stop)
                result_shape = (item.stop - item.start, self.shape[1])
            else:
                raise NotImplementedError("Only 2 or less dimensions array is supported.")
            result = Storage(result_data, result_shape, self.dtype)
        else:
            result = self.data.to_host()[item]
        return result

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
        return self.data.size()

    def __bool__(self):
        if len(self) == 1:
            result = self.data.to_host()[0] != 0
        else:
            raise NotImplementedError("Logic value of array is ambiguous.")
        return result

    def detach(self):
        if isinstance(self.data, trtc.DVVector.DVRange):
            if self.dtype is Storage.FLOAT:
                elem_cls = 'double'
            elif self.dtype in Storage.INT:
                elem_cls = 'int64_t'
            else:
                raise NotImplementedError()

            data = trtc.device_vector(elem_cls, self.data.size())

            trtc.Copy(self.data, data)
            self.data = data

    def download(self, target):
        self.detach()
        target[:] = np.reshape(self.data.to_host(), self.shape)

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

        data = array.astype(dtype).copy()
        result = Storage(data, array.shape, dtype)
        return result

    def floor(self, other=None):
        if other is None:
            impl.floor(self.data)
        else:
            impl.floor_out_of_place(self.data, other.data)
        return self

    def product(self, multiplicand, multiplier):
        impl.multiply_out_of_place(self, multiplicand, multiplier)
        return self

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

    def upload(self, data):
        self.data = trtc.device_vector_from_numpy(data.flatten())

    def write_row(self, i, row):
        start = self.shape[1] * i
        stop = start + self.shape[1]
        trtc.Copy(row.data, self.data.range(start, stop))

    def fill(self, value):
        trtc.Fill(self.data, value)
        return self
