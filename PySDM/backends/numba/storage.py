"""
Created at 30.05.2020
"""

import numpy as np
from ._maths_methods import MathsMethods


class Storage:

    FLOAT = np.float64
    INT = np.int64

    def __init__(self, data, shape, dtype):
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, item):
        start = item.start or 0
        if isinstance(item, slice):
            dim = len(self.shape)
            if dim == 1:
                stop = item.stop or len(self)
                result_data = self.data[item]
                result_shape = (stop - start,)
            elif dim == 2:
                stop = item.stop or self.shape[1]
                result_data = self.data[item]
                result_shape = (stop - start, self.shape[1])
            else:
                raise NotImplementedError("Only 2 or less dimensions array is supported.")
            result = Storage(result_data, result_shape, self.dtype)
        else:
            result = self.data[item]
        return result

    def __add__(self, other):
        raise NotImplementedError("Use +=")

    def __iadd__(self, other):
        MathsMethods.add(self.data, other.data)
        return self

    def __sub__(self, other):
        raise NotImplementedError("Use -=")

    def __isub__(self, other):
        MathsMethods.subtract(self.data, other.data)
        return self

    def __mul__(self, other):
        raise NotImplementedError("Use *=")

    def __imul__(self, other):
        if hasattr(other, 'data'):
            MathsMethods.multiply(self.data, other.data)
        else:
            MathsMethods.multiply(self.data, other)
        return self

    def __mod__(self, other):
        raise NotImplementedError("Use %=")

    def __imod__(self, other):
        # TODO
        MathsMethods.row_modulo(self.data, other.data)
        return self

    def __pow__(self, other):
        raise NotImplementedError("Use **=")

    def __ipow__(self, other):
        MathsMethods.power(self.data, other.data)
        return self

    def __len__(self):
        return self.data.size

    def __bool__(self):
        if len(self) == 1:
            result = bool(self.data[0] != 0)
        else:
            raise NotImplementedError("Logic value of array is ambiguous.")
        return result

    def detach(self):
        if self.data.base is not None:
            self.data = np.array(self.data)

    def download(self, target):
        np.copyto(target, self.data, casting='safe')

    @staticmethod
    def empty(shape, dtype):
        if dtype in (float, Storage.FLOAT):
            data = np.full(shape, -1., dtype=Storage.FLOAT)
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT):
            data = np.full(shape, -1, dtype=Storage.INT)
            dtype = Storage.INT
        else:
            raise NotImplementedError()

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
            MathsMethods.floor(self.data)
        else:
            MathsMethods.floor_out_of_place(self.data, other.data)
        return self

    def product(self, multiplicand, multiplier):
        MathsMethods.multiply_out_of_place(self, multiplicand, multiplier)
        return self

    def read_row(self, i):
        result = Storage(self.data[i, :], *self.shape[1:], self.dtype)
        return result

    def shuffle(self, generator=None, parts=None):
        raise NotImplementedError()

    def urand(self, generator=None):
        raise NotImplementedError()

    def to_ndarray(self):
        return self.data.copy()

    def upload(self, data):
        np.copyto(self.data, data, casting='safe')

    def write_row(self, i, row):
        self.data[i, :] = row

    def fill(self, value):
        self.data[:] = value
        return self
