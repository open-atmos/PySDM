import abc
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional, Sized, Tuple, Type, Union

import numpy as np
from typing_extensions import TypeGuard

from PySDM.storages.common.random import Random

ShapeType = Union[int, Tuple[int, ...]]


class StorageSignature(NamedTuple):
    """groups items defining a storage"""

    data: np.ndarray
    shape: ShapeType
    dtype: Type


_OtherType = Union["Storage", np.ndarray]


class MathMagicMethodsMixin:
    def __pow__(self, other: _OtherType):
        raise TypeError("Use **=")

    @abc.abstractmethod
    def __ipow__(self, other: _OtherType):
        raise NotImplementedError()

    def __mod__(self, other: _OtherType):
        raise TypeError("Use %=")

    @abc.abstractmethod
    def __imod__(self, other: _OtherType):
        raise NotImplementedError()

    def __truediv__(self, other: _OtherType):
        raise TypeError("Use /=")

    @abc.abstractmethod
    def __itruediv__(self, other: _OtherType):
        raise NotImplementedError()

    def __mul__(self, other: _OtherType):
        raise TypeError("Use *=")

    @abc.abstractmethod
    def __imul__(self, other: _OtherType):
        raise NotImplementedError()

    def __sub__(self, other: _OtherType):
        raise TypeError("Use -=")

    @abc.abstractmethod
    def __isub__(self, other: _OtherType):
        raise NotImplementedError()

    def __add__(self, other: _OtherType):
        raise TypeError("Use +=")

    @abc.abstractmethod
    def __iadd__(self, other: _OtherType):
        raise NotImplementedError()


class StorageOperationsMixin(Sequence):
    @abc.abstractmethod
    def to_ndarray(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def upload(self, data: np.ndarray):
        raise NotImplementedError()

    @abc.abstractmethod
    def download(self, target: np.ndarray, reshape: bool = False):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _get_empty_data(cls, shape: ShapeType, dtype: Type) -> StorageSignature:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _get_data_from_ndarray(cls, array: np.ndarray) -> StorageSignature:
        raise NotImplementedError()

    @abc.abstractmethod
    def amin(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def floor(self, other: Optional[_OtherType] = None) -> "Storage":
        raise NotImplementedError()

    @abc.abstractmethod
    def product(self, multiplicand: _OtherType, multiplier: _OtherType) -> "Storage":
        raise NotImplementedError()

    @abc.abstractmethod
    def ratio(self, dividend: _OtherType, divisor: _OtherType) -> "Storage":
        raise NotImplementedError()

    @abc.abstractmethod
    def divide_if_not_zero(self, divisor: _OtherType) -> "Storage":
        raise NotImplementedError()

    @abc.abstractmethod
    def sum(self, arg_a: _OtherType, arg_b: _OtherType) -> "Storage":
        raise NotImplementedError()

    @abc.abstractmethod
    def ravel(self, other: _OtherType) -> None:
        raise NotImplementedError()


class Storage(MathMagicMethodsMixin, StorageOperationsMixin):
    INT: Type
    FLOAT: Type
    BOOL: Type

    data: np.ndarray
    shape: ShapeType
    dtype: Type

    def __init__(self, signature: StorageSignature):
        self.data = signature.data
        self.shape = (
            (signature.shape,) if isinstance(signature.shape, int) else signature.shape
        )
        self.dtype = signature.dtype

    def __len__(self):
        return next(iter(self.shape))

    @classmethod
    def empty(cls, shape: ShapeType, dtype: Type) -> "Storage":
        return cls(cls._get_empty_data(shape, dtype))

    @classmethod
    def from_ndarray(cls, array: np.ndarray) -> "Storage":
        return cls(cls._get_data_from_ndarray(array))

    def urand(self, generator: Random) -> None:
        generator(self)

    @property
    def signature(self) -> StorageSignature:
        return StorageSignature(self.data, self.shape, self.dtype)

    @staticmethod
    def is_storage(obj: Any) -> TypeGuard["Storage"]:
        return isinstance(obj, Storage)

    def all(self) -> bool:
        return self.data.all()


class Index(Sized):
    @classmethod
    @abc.abstractmethod
    def identity_index(cls, length: int) -> "Index":
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_index(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def sort_by_key(self, keys) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def shuffle(self, temporary, parts=None) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_zero_n_or_flagged(self, indexed_storage: "Indexed") -> None:
        raise NotImplementedError()


class Indexed(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def indexed(cls, idx: "Index", storage: Storage) -> "Indexed":
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def indexed_and_empty(
        cls, idx: "Index", shape: ShapeType, dtype: Type
    ) -> "Indexed":
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def indexed_from_ndarray(cls, idx: "Index", array: np.ndarray) -> "Indexed":
        raise NotImplementedError()


class Pairwise(abc.ABC):
    @abc.abstractmethod
    def distance(self, other, is_first_in_pair) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def max(self, other, is_first_in_pair) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def min(self, other, is_first_in_pair) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def sort(self, other, is_first_in_pair) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def sum(self, other, is_first_in_pair) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def multiply(self, other, is_first_in_pair) -> None:
        raise NotImplementedError()
