from typing import Callable, Type

import numpy as np

from PySDM.storages.common.storage import Storage, StorageSignature


def get_data_from_ndarray(
    array: np.ndarray,
    storage_class: Type[Storage],
    copy_fun: Callable[[np.ndarray], np.ndarray],
) -> StorageSignature:
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
