"""
common code for storage classes
"""

from abc import abstractmethod
from collections import namedtuple
from typing import Type


class StorageSignature(namedtuple("StorageSignature", ("data", "shape", "dtype"))):
    """groups items defining a storage"""

    __slots__ = ()


class StorageBase:
    def __init__(self, signature: StorageSignature):
        self.data = signature.data
        self.shape = (
            (signature.shape,) if isinstance(signature.shape, int) else signature.shape
        )
        self.dtype = signature.dtype
        self.backend = None

    def __len__(self):
        return self.shape[0]

    def __pow__(self, other):
        raise TypeError("Use **=")

    def __mod__(self, other):
        raise TypeError("Use %=")

    def __truediv__(self, other):
        raise TypeError("Use /=")

    def __mul__(self, other):
        raise TypeError("Use *=")

    def __sub__(self, other):
        raise TypeError("Use -=")

    def __add__(self, other):
        raise TypeError("Use +=")

    @abstractmethod
    def to_ndarray(self):
        raise NotImplementedError()

    @abstractmethod
    def urand(self, generator):
        raise NotImplementedError()

    @abstractmethod
    def upload(self, data):
        raise NotImplementedError()

    @abstractmethod
    def fill(self, other):
        raise NotImplementedError()


def get_data_from_ndarray(array, storage_class: Type[StorageBase], copy_fun):
    if str(array.dtype).startswith("int"):
        dtype = storage_class.INT
    elif str(array.dtype).startswith("float"):
        dtype = storage_class.FLOAT
    elif str(array.dtype).startswith("bool"):
        dtype = storage_class.BOOL
    else:
        raise NotImplementedError()

    data = copy_fun(array.astype(dtype))

    return StorageSignature(data, array.shape, dtype)


def empty(shape, dtype, storage_class: Type[StorageBase]):
    return storage_class(storage_class._get_empty_data(shape, dtype))
