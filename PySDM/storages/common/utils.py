from typing import Callable, Type, Union, overload

import numpy as np

from PySDM.storages.common.storage import Storage, StorageSignature
from PySDM.storages.thrust_rtc.conf import trtc

DataType = Union[np.ndarray, trtc.DVVector]


@overload
def get_data_from_ndarray(
    array: np.ndarray,
    storage_class: Type[Storage],
    copy_fun: Callable[[np.ndarray], np.ndarray],
) -> StorageSignature[np.ndarray]:
    ...


@overload
def get_data_from_ndarray(
    array: np.ndarray,
    storage_class: Type[Storage],
    copy_fun: Callable[[np.ndarray], trtc.DVVector],
) -> StorageSignature[trtc.DVVector]:
    ...


def get_data_from_ndarray(
    array: np.ndarray,
    storage_class: Type[Storage],
    copy_fun: Callable[[np.ndarray], DataType],
) -> StorageSignature:
    """
    Creates a storage signature from a numpy array.

    Parameters
    ----------
    array : np.ndarray
        numpy array to be converted to a storage signature
    storage_class : Type[Storage]
        storage class to be used for type deduction
    copy_fun : Callable[[np.ndarray], DataType]
        function to be used for copying the data from the numpy array to the storage data

    """
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
