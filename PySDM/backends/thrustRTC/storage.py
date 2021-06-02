import numpy as np

from PySDM.backends.thrustRTC.conf import trtc
from PySDM.backends.thrustRTC.impl import storage_impl as impl
from PySDM.backends.thrustRTC.impl.precision_resolver import PrecisionResolver


class Storage:

    FLOAT = PrecisionResolver.get_np_dtype()
    INT = np.int64
    BOOL = np.bool_

    def __init__(self, data, shape, dtype):
        self.data = data
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype

    def __getitem__(self, item):
        dim = len(self.shape)
        if isinstance(item, slice):
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
        elif isinstance(item, tuple) and dim == 2 and isinstance(item[0], int) and isinstance(item[1], slice):
            assert item[1].start is None or item[1].start == 0
            assert item[1].stop is None or item[1].stop == self.shape[1]
            assert item[1].step is None or item[1].step == 1
            result_data = self.data.range(self.shape[1] * item[0], self.shape[1] * (item[0] + 1))
            result = Storage(result_data, (*self.shape[1:],), self.dtype)
        else:
            result = self.to_ndarray()[item]
        return result

    def __setitem__(self, key, value):
        if hasattr(value, 'data') and hasattr(value, 'shape') and len(value.shape) != 0:
            if isinstance(value, np.ndarray):
                vector = trtc.device_vector_from_numpy(value)
                trtc.Copy(vector, self.data)
            else:
                trtc.Copy(value.data, self.data)
        else:
            if isinstance(value, int):
                dvalue = trtc.DVInt64(value)
            elif isinstance(value, float):
                dvalue = PrecisionResolver.get_floating_point(value)
            else:
                raise TypeError("Only Storage, int and float are supported.")
            trtc.Fill(self.data, dvalue)
        return self

    def __add__(self, other):
        raise TypeError("Use +=")

    def __iadd__(self, other):
        impl.add(self, other)
        return self

    def __sub__(self, other):
        raise TypeError("Use -=")

    def __isub__(self, other):
        impl.subtract(self, other)
        return self

    def __mul__(self, other):
        raise TypeError("Use *=")

    def __imul__(self, other):
        impl.multiply(self, other)
        return self

    def __truediv__(self, other):
        raise TypeError("Use /=")

    def __itruediv__(self, other):
        impl.truediv(self, other)
        return self

    def __mod__(self, other):
        raise TypeError("Use %=")

    def __imod__(self, other):
        impl.row_modulo(self, other)
        return self

    def __pow__(self, other):
        raise TypeError("Use **=")

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

    def _to_host(self):
        if isinstance(self.data, trtc.DVVector.DVRange):
            if self.dtype is Storage.FLOAT:
                elem_cls = PrecisionResolver.get_C_type()
            elif self.dtype is Storage.INT:
                elem_cls = 'int64_t'
            elif self.dtype is Storage.BOOL:
                elem_cls = 'bool'
            else:
                raise NotImplementedError()

            data = trtc.device_vector(elem_cls, self.data.size())

            trtc.Copy(self.data, data)
        else:
            data = self.data
        return data.to_host()

    def amin(self):
        return impl.amin(self.data)

    def all(self):
        assert self.dtype is Storage.BOOL
        return self.amin()

    def download(self, target, reshape=False):
        shape = target.shape if reshape else self.shape
        target[:] = np.reshape(self._to_host(), shape)

    @staticmethod
    def _get_empty_data(shape, dtype):
        if dtype in (float, Storage.FLOAT):
            elem_cls = PrecisionResolver.get_C_type()
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT):
            elem_cls = 'int64_t'
            dtype = Storage.INT
        elif dtype in (bool, Storage.BOOL):
            elem_cls = 'bool'
            dtype = Storage.BOOL
        else:
            raise NotImplementedError

        data = trtc.device_vector(elem_cls, int(np.prod(shape)))
        return data, shape, dtype

    @staticmethod
    def empty(shape, dtype):
        result = Storage(*Storage._get_empty_data(shape, dtype))
        return result

    @staticmethod
    def _get_data_from_ndarray(array):
        if str(array.dtype).startswith('int'):
            dtype = Storage.INT
        elif str(array.dtype).startswith('float'):
            dtype = Storage.FLOAT
        elif str(array.dtype).startswith('bool'):
            dtype = Storage.BOOL
        else:
            raise NotImplementedError()

        data = trtc.device_vector_from_numpy(array.astype(dtype).ravel())
        return data, array.shape, dtype

    @staticmethod
    def from_ndarray(array):
        result = Storage(*Storage._get_data_from_ndarray(array))
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

    def ratio(self, dividend, divisor):
        impl.divide_out_of_place(self, dividend, divisor)
        return self

    def ravel(self, other):
        if isinstance(other, Storage):
            trtc.Copy(other.data, self.data)
        else:
            self.data = trtc.device_vector_from_numpy(other.ravel())

    def to_ndarray(self):
        result = self._to_host()
        result = np.reshape(result, self.shape)
        return result

    def urand(self, generator=None):
        generator(self)

    def upload(self, data):
        trtc.Copy(
            trtc.device_vector_from_numpy(data.astype(self.dtype).ravel()),
            self.data
        )

