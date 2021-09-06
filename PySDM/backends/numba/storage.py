import numpy as np
from PySDM.backends.numba.impl import storage_impl as impl


class Storage:

    FLOAT = np.float64
    INT = np.int64
    BOOL = np.bool_

    def __init__(self, data, shape, dtype):
        self.data = data
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype

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
                raise NotImplementedError("Only 2 or less dimensions array is supported.")
            if stop > self.data.shape[0]:
                raise IndexError(f"requested a slice ({start}:{stop}) of Storage with first dim of length {self.data.shape[0]}")
            result = Storage(result_data, result_shape, self.dtype)
        elif isinstance(item, tuple) and dim == 2 and isinstance(item[1], slice):
            result = Storage(self.data[item[0]], (*self.shape[1:],), self.dtype)
        else:
            result = self.data[item]
        return result

    def __setitem__(self, key, value):
        if hasattr(value, 'data'):
            self.data[key] = value.data
        else:
            self.data[key] = value
        return self

    def __add__(self, other):
        raise TypeError("Use +=")

    def __iadd__(self, other):
        if isinstance(other, Storage):
            impl.add(self.data, other.data)
        else:
            impl.add(self.data, other)
        return self

    def __sub__(self, other):
        raise TypeError("Use -=")

    def __isub__(self, other):
        impl.subtract(self.data, other.data)
        return self

    def __mul__(self, other):
        raise TypeError("Use *=")

    def __imul__(self, other):
        if hasattr(other, 'data'):
            impl.multiply(self.data, other.data)
        else:
            impl.multiply(self.data, other)
        return self

    def __truediv__(self, other):
        raise TypeError("Use /=")

    def __itruediv__(self, other):
        if hasattr(other, 'data'):
            self.data[:] /= other.data[:]
        else:
            self.data[:] /= other
        return self

    def __mod__(self, other):
        raise TypeError("Use %=")

    def __imod__(self, other):
        impl.row_modulo(self.data, other.data)
        return self

    def __pow__(self, other):
        raise TypeError("Use **=")

    def __ipow__(self, other):
        impl.power(self.data, other)
        return self

    def __len__(self):
        return self.shape[0]

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
        np.copyto(target, data, casting='safe')

    @staticmethod
    def _get_empty_data(shape, dtype):
        if dtype in (float, Storage.FLOAT):
            data = np.full(shape, -1., dtype=Storage.FLOAT)
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT):
            data = np.full(shape, -1, dtype=Storage.INT)
            dtype = Storage.INT
        elif dtype in (bool, Storage.BOOL):
            data = np.full(shape, -1, dtype=Storage.BOOL)
            dtype = Storage.BOOL
        else:
            raise NotImplementedError()

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

        data = array.astype(dtype).copy()

        return data, array.shape, dtype

    def amin(self):
        return impl.amin(self.data)

    def all(self):
        return self.data.all()

    @staticmethod
    def from_ndarray(array):
        result = Storage(*Storage._get_data_from_ndarray(array))
        return result

    def floor(self, other=None):
        if other is None:
            impl.floor(self.data)
        else:
            impl.floor_out_of_place(self.data, other.data)
        return self

    def product(self, multiplicand, multiplier):
        if hasattr(multiplier, 'data'):
            impl.multiply_out_of_place(self.data, multiplicand.data, multiplier.data)
        else:
            impl.multiply_out_of_place(self.data, multiplicand.data, multiplier)
        return self

    def ratio(self, dividend, divisor):
        impl.divide_out_of_place(self.data, dividend.data, divisor.data)
        return self

    def sum(self, a, b):
        impl.sum_out_of_place(self.data, a.data, b.data)
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
        np.copyto(self.data, data, casting='safe')
