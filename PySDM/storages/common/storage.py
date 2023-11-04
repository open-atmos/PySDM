import abc
from collections.abc import Sequence
from numbers import Number
from typing import Any, NamedTuple, Optional, Protocol, Sized, Tuple, Type, Union

import numpy as np
from typing_extensions import TypeGuard

from PySDM.storages.common.random import Random
from PySDM.storages.thrust_rtc.conf import trtc

ShapeType = Union[int, Tuple[int, ...]]
DataType = Union[np.ndarray, trtc.DVVector]


class StorageSignature(NamedTuple):
    """Groups items defining a storage"""

    #: storage data
    data: DataType
    #: storage shape
    shape: ShapeType
    #: storage dtype
    dtype: Type


_OtherType = Union["Storage", np.ndarray, Number]


class MathMagicMethodsMixin:
    """Mixin for math magic methods"""

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
    """Mixin for storage operations"""

    @abc.abstractmethod
    def to_ndarray(self, *args, **kwargs) -> np.ndarray:
        """Convert storage to numpy array"""
        raise NotImplementedError()

    @abc.abstractmethod
    def upload(self, data: np.ndarray) -> None:
        """
        Upload data to storage

        Parameters
        ----------
        data : np.ndarray
            data to be uploaded

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def download(self, target: np.ndarray, reshape: bool = False):
        """
        Download data from storage

        Parameters
        ----------
        target : np.ndarray
            target array for downloaded data
        reshape : bool
            if True, reshape target array to storage shape
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _get_empty_data(cls, shape: ShapeType, dtype: Type) -> StorageSignature:
        """
        Get empty storage data signature

        Parameters
        ----------
        shape : ShapeType
            storage shape
        dtype : Type
            storage dtype

        Returns
        -------
        StorageSignature
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _get_data_from_ndarray(cls, array: np.ndarray) -> StorageSignature:
        """
        Get storage data signature from numpy array

        Parameters
        ----------
        array : np.ndarray
            numpy array to be converted to a storage signature

        Returns
        -------
        StorageSignature
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def amin(self):
        """
        Get minimum value of storage

        Returns
        -------
        minimum value
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def floor(self, other: Optional[_OtherType] = None) -> "Storage":
        """
        Get floor of storage

        Parameters
        ----------
        other : Optional[_OtherType]
            if not None, floor of other is returned

        Returns
        -------
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def product(self, multiplicand: _OtherType, multiplier: _OtherType) -> "Storage":
        """
        Get product of two storages

        Parameters
        ----------
        multiplicand : _OtherType
            multiplicand
        multiplier : _OtherType
            multiplier

        Returns
        -------
        product
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def ratio(self, dividend: _OtherType, divisor: _OtherType) -> "Storage":
        """
        Get ratio of two storages

        Parameters
        ----------
        dividend : _OtherType
            dividend
        divisor : _OtherType
            divisor

        Returns
        -------
        ratio
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def divide_if_not_zero(self, divisor: _OtherType) -> "Storage":
        """
        Get division of storage by divisor if divisor is not zero

        Parameters
        ----------
        divisor : _OtherType
            divisor

        Returns
        -------
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sum(self, arg_a: _OtherType, arg_b: _OtherType) -> "Storage":
        """
        Get sum of two storages

        Parameters
        ----------
        arg_a : _OtherType
            first summand
        arg_b : _OtherType
            second summand

        Returns
        -------
        sum
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def ravel(self, other: _OtherType) -> None:
        """
        Ravel other storage to self

        Parameters
        ----------
        other : _OtherType
            storage to be raveled

        Returns
        -------
        None
        """
        raise NotImplementedError()


class Storage(MathMagicMethodsMixin, StorageOperationsMixin, Sized):
    """Storage class"""

    #: the integer type used
    INT: Type
    #: the float type used
    FLOAT: Type
    #: the boolean type used
    BOOL: Type

    #: the internal storage data
    data: np.ndarray
    #: the storage shape
    shape: ShapeType
    #: the storage dtype
    dtype: Type

    def __init__(self, signature: StorageSignature):
        """
        Initialize storage

        If the shape is integer, it is converted to a tuple of length 1.

        Parameters
        ----------
        signature : StorageSignature
            storage signature

        Returns
        -------
        None
        """
        self.data = signature.data
        self.shape = (
            (signature.shape,) if isinstance(signature.shape, int) else signature.shape
        )
        self.dtype = signature.dtype

    def __len__(self) -> int:
        """
        Get length of storage

        Returns
        -------
        length
        """
        return next(iter(self.shape))

    @classmethod
    def empty(cls, shape: ShapeType, dtype: Type) -> "Storage":
        """
        Get empty storage

        Parameters
        ----------
        shape : ShapeType
            storage shape
        dtype : Type
            storage dtype

        Returns
        -------
        empty storage
        """
        return cls(cls._get_empty_data(shape, dtype))

    @classmethod
    def from_ndarray(cls, array: np.ndarray) -> "Storage":
        """
        Get storage from numpy array

        Parameters
        ----------
        array : np.ndarray
            numpy array to be converted to a storage

        Returns
        -------
        storage
        """
        return cls(cls._get_data_from_ndarray(array))

    def urand(self, generator: Random) -> None:
        """
        Fill storage with random numbers

        Parameters
        ----------
        generator : Random
            random number generator

        Returns
        -------
        None
        """
        generator(self)

    @property
    def signature(self) -> StorageSignature:
        """
        Get storage signature

        Returns
        -------
        storage signature
        """
        return StorageSignature(self.data, self.shape, self.dtype)

    @staticmethod
    def is_storage(obj: Any) -> TypeGuard["Storage"]:
        """
        Type checks if a given object is a storage

        Parameters
        ----------
        obj : Any
            object to be checked

        Returns
        -------
        True if object is a storage, False otherwise
        """
        return isinstance(obj, Storage)

    def all(self) -> bool:
        """
        Test whether all array elements along a given axis evaluate to True.

        Returns
        -------
        all : bool
        """
        return self.data.all()


class Index(Storage):
    """Index class"""

    @classmethod
    @abc.abstractmethod
    def identity_index(cls, length: int) -> "Index":
        """
        Get identity index

        Parameters
        ----------
        length : int
            length of identity index

        Returns
        -------
        identity index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_index(self) -> None:
        """
        Reset index

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sort_by_key(self, keys) -> None:
        """
        Sort index by keys

        Parameters
        ----------
        keys : Storage
            keys to sort by

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def shuffle(self, temporary: Storage, parts: Optional[Storage] = None) -> None:
        """
        Shuffle index

        Parameters
        ----------
        temporary : Storage
            temporary storage

        parts : Optional[Storage]
            parts to shuffle

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_zero_n_or_flagged(self, indexed_storage: "Indexed") -> None:
        """
        Remove zero n or flagged elements from index

        Parameters
        ----------
        indexed_storage : Indexed
            an indexed storage to remove elements from

        Returns
        -------
        None
        """
        raise NotImplementedError()


class Indexed(Storage):
    """Indexed class"""

    #: the index
    idx: Index

    @classmethod
    @abc.abstractmethod
    def indexed(cls, idx: "Index", storage: Storage) -> "Indexed":
        """
        Get indexed storage

        Parameters
        ----------
        idx : Index
            index
        storage : Storage
            storage to be indexed

        Returns
        -------
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def indexed_and_empty(
        cls, idx: "Index", shape: ShapeType, dtype: Type
    ) -> "Indexed":
        """
        Get indexed and empty storage

        Parameters
        ----------
        idx : Index
            index
        shape : ShapeType
            storage shape
        dtype : Type
            storage dtype

        Returns
        -------
        a new indexed and empty storage
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def indexed_from_ndarray(cls, idx: "Index", array: np.ndarray) -> "Indexed":
        """
        Get indexed storage from numpy array

        Parameters
        ----------
        idx : Index
            index
        array : np.ndarray
            numpy array to be converted to a storage

        Returns
        -------
        indexed storage
        """
        raise NotImplementedError()


class PairIndicator(Sized):
    """Pair indicator class"""

    @abc.abstractmethod
    def update(self, cell_start, cell_idx, cell_id):
        """
        Update pair indicator

        Parameters
        ----------
        cell_start : Storage
            cell start
        cell_idx : Storage
            cell index
        cell_id : Indexed
            cell id
        """
        raise NotImplementedError()


class Pairwise(Storage):
    """Pairwise Storage class"""

    @abc.abstractmethod
    def distance(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
        """
        Compute distance between two indexed storages pair-wise

        Parameters
        ----------
        other : Indexed
            other indexed storage
        is_first_in_pair : PairIndicator
            pair indicator

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def max(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
        """
        Compute max between two indexed storages pair-wise

        Parameters
        ----------
        other : Indexed
            other indexed storage
        is_first_in_pair : PairIndicator
            pair indicator

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def min(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
        """
        Compute min between two indexed storages pair-wise

        Parameters
        ----------
        other : Indexed
            other indexed storage
        is_first_in_pair : PairIndicator
            pair indicator

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sort(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
        """
        Sort two indexed storages pair-wise

        Parameters
        ----------
        other : Indexed
            other indexed storage
        is_first_in_pair : PairIndicator
            pair indicator

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sum(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
        """
        Compute sum between two indexed storages pair-wise

        Parameters
        ----------
        other : Indexed
            other indexed storage
        is_first_in_pair : PairIndicator
            pair indicator

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def multiply(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
        """
        Compute multiplication between two indexed storages pair-wise

        Parameters
        ----------
        other : Indexed
            other indexed storage
        is_first_in_pair : PairIndicator
            pair indicator

        Returns
        -------
        None
        """
        raise NotImplementedError()


class StoragesHolder(Protocol):
    """
    A protocol for a class that holds storages.

    This protocol is used to type check the storages' holder.
    """

    #: The base storage class.
    Storage: Type[Storage]
    #: The index class.
    Index: Type[Index]
    #: The base indexed class.
    IndexedStorage: Type[Indexed]
    #: The pairwise storage class.
    PairwiseStorage: Type[Pairwise]
    #: The pair indicator class.
    PairIndicator: Type[PairIndicator]
    #: Random number generator.
    Random: Type[Random]
