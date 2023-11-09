"""
The backend base classes

The backend base classes are used to define the interface of the backends.
The backends are responsible for the actual implementation of the storage operations.
"""
from typing import Protocol, Union

import numpy as np

from PySDM.storages.common.storage import Index, PairIndicator, Storage
from PySDM.storages.thrust_rtc.conf import trtc

DataType = Union[np.ndarray, trtc.DVVector]


class Backend(Protocol):
    """
    The backend base class.

    The backend base class is used to define the interface of the backends.
    """


class IndexBackend(Backend):
    """ "
    Backend for the index storage.
    """

    @staticmethod
    def identity_index(idx: DataType) -> None:
        """
        Sets the index to the identity index.

        Parameters
        ----------
        idx : DataType
            index data to be set to the identity index

        Returns
        -------
        None
        """

    @staticmethod
    def shuffle_global(idx: DataType, length: int, u01: DataType) -> None:
        """
        Shuffles the index globally.

        Parameters
        ----------
        idx : DataType
            index data to be shuffled
        length : int
            length of the index
        u01 : DataType
            temporary data to be used for the shuffling

        Returns
        -------
        None
        """

    @staticmethod
    def shuffle_local(idx: DataType, u01: DataType, cell_start: DataType) -> None:
        """
        Shuffles the index locally.

        Parameters
        ----------
        idx : DataType
            index data to be shuffled
        u01 : DataType
            temporary data to be used for the shuffling
        cell_start : DataType
            cell start to be used for the shuffling

        Returns
        -------
        None
        """

    @staticmethod
    def sort_by_key(idx: Storage, attr: Storage) -> None:
        """
        Sorts the index by the given attribute.

        Parameters
        ----------
        idx : Index
            index data to be sorted
        attr : Storage
            attribute data to be used for sorting

        Returns
        -------
        None
        """

    @staticmethod
    def remove_zero_n_or_flagged(
        multiplicity: DataType, idx: DataType, length: int
    ) -> int:
        """
        Removes particles with zero n or flagged.

        Parameters
        ----------
        multiplicity : DataType
            multiplicity data to be used for the removal
        idx : DataType
            an index to be used for the removal
        length : int
            length of the index

        Returns
        -------
        int
            new length of the index
        """


class PairBackend(Backend):
    @staticmethod
    def distance_pair(
        data_out: Storage, data_in: Storage, is_first_in_pair: PairIndicator, idx: Index
    ) -> None:
        """
        Calculates the distance between pairs of particles.

        Parameters
        ----------
        data_out : Storage
            output storage
        data_in : Storage
            input storage
        is_first_in_pair : PairIndicator
            a storage indicating whether a particle is the first in a pair
        idx : Index
            an index to be used for the calculation

        Returns
        -------
        None
        """

    @staticmethod
    def max_pair(
        data_out: Storage, data_in: Storage, is_first_in_pair: PairIndicator, idx: Index
    ) -> None:
        """
        Calculates the maximum of pairs of particles.

        Parameters
        ----------
        data_out : Storage
            output storage
        data_in : Storage
            input storage
        is_first_in_pair : PairIndicator
            a storage indicating whether a particle is the first in a pair
        idx : Index
            an index to be used for the calculation

        Returns
        -------
        None
        """

    @staticmethod
    def min_pair(
        data_out: Storage, data_in: Storage, is_first_in_pair: PairIndicator, idx: Index
    ) -> None:
        """
        Calculates the minimum of pairs of particles.

        Parameters
        ----------
        data_out : Storage
            output storage
        data_in : Storage
            input storage
        is_first_in_pair : PairIndicator
            a storage indicating whether a particle is the first in a pair
        idx : Index
            an index to be used for the calculation

        Returns
        -------
        None
        """

    @staticmethod
    def sort_pair(
        data_out: Storage, data_in: Storage, is_first_in_pair: PairIndicator, idx: Index
    ) -> None:
        """
        Sorts pairs of particles.

        Parameters
        ----------
        data_out : Storage
            output storage
        data_in : Storage
            input storage
        is_first_in_pair : PairIndicator
            a storage indicating whether a particle is the first in a pair
        idx : Index
            an index to be used for the calculation

        Returns
        -------
        None
        """

    @staticmethod
    def sum_pair(
        data_out: Storage, data_in: Storage, is_first_in_pair: PairIndicator, idx: Index
    ) -> None:
        """
        Calculates the sum of pairs of particles.

        Parameters
        ----------
        data_out : Storage
            output storage
        data_in : Storage
            input storage
        is_first_in_pair : PairIndicator
            a storage indicating whether a particle is the first in a pair
        idx : Index
            an index to be used for the calculation

        Returns
        -------
        None
        """

    @staticmethod
    def multiply_pair(
        data_out: Storage, data_in: Storage, is_first_in_pair: PairIndicator, idx: Index
    ) -> None:
        """
        Calculates the product of pairs of particles.

        Parameters
        ----------
        data_out : Storage
            output storage
        data_in : Storage
            input storage
        is_first_in_pair : PairIndicator
            a storage indicating whether a particle is the first in a pair
        idx : Index
            an index to be used for the calculation

        Returns
        -------
        None
        """
