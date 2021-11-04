from collections import namedtuple
from typing import Type
from abc import abstractmethod

StorageSignature = namedtuple("StorageSignature", ('data', 'shape', 'dtype'))


class StorageBase:
    def __init__(self, signature: StorageSignature):
        self.data = signature.data
        self.shape = (signature.shape,) if isinstance(signature.shape, int) else signature.shape
        self.dtype = signature.dtype

    @abstractmethod
    def to_ndarray(self):
        raise NotImplementedError()

    @abstractmethod
    def urand(self, generator=None):
        raise NotImplementedError()

    @abstractmethod
    def upload(self, data):
        raise NotImplementedError()


def get_data_from_ndarray(array, storage_class: Type[StorageBase], copy_fun):
    if str(array.dtype).startswith('int'):
        dtype = storage_class.INT
    elif str(array.dtype).startswith('float'):
        dtype = storage_class.FLOAT
    elif str(array.dtype).startswith('bool'):
        dtype = storage_class.BOOL
    else:
        raise NotImplementedError()

    data = copy_fun(array.astype(dtype))

    return StorageSignature(data, array.shape, dtype)


def empty(shape, dtype, storage_class: Type[StorageBase]):
    return storage_class(storage_class._get_empty_data(shape, dtype))
