"""
Created at 10.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc
import numpy as np


class StorageMethods:
    # TODO check static For
    storage = trtc.DVVector.DVVector
    integer = np.int64
    double = np.float64

    @staticmethod
    def array(shape, dtype):
        if dtype in (float, StorageMethods.double):
            elem_cls = 'double'
            elem_dtype = StorageMethods.double
        elif dtype in (int, StorageMethods.integer):
            elem_cls = 'int64_t'
            elem_dtype = StorageMethods.integer
        else:
            raise NotImplementedError

        data = trtc.device_vector(elem_cls, int(np.prod(shape)))
        # TODO: trtc.Fill(data, trtc.DVConstant(np.nan))

        StorageMethods.__equip(data, shape, elem_dtype)
        return data

    @staticmethod
    def download(backend_data, numpy_target):
        if isinstance(backend_data, StorageMethods.storage):
            numpy_target[:] = np.reshape(backend_data.to_host(), backend_data.shape)
        else:
            numpy_target[:] = StorageMethods.to_ndarray(backend_data)

    @staticmethod
    def from_ndarray(array):
        shape = array.shape

        if str(array.dtype).startswith('int'):
            dtype = StorageMethods.integer
        elif str(array.dtype).startswith('float'):
            dtype = StorageMethods.double
        else:
            raise NotImplementedError

        if array.ndim > 1:
            array = array.astype(dtype).flatten()
        else:
            array = array.astype(dtype)

        result = trtc.device_vector_from_numpy(array)

        StorageMethods.__equip(result, shape, dtype)
        return result

    @staticmethod
    def range(array, start=0, stop=None):
        if stop is None:
            stop = array.shape[0]
        dim = len(array.shape)
        if dim == 1:
            result = array.range(start, stop)
            new_shape = (stop - start, )
        elif dim == 2:
            result = array.range(array.shape[1] * start, array.shape[1] * stop)
            new_shape = (stop - start, array.shape[1])
        else:
            raise NotImplementedError("Only 3 or more dimensions array is supported.")
        StorageMethods.__equip(result, shape=new_shape, dtype=array.dtype)
        return result

    @staticmethod
    def read_row(array, i):
        row_length = array.shape[1]
        start = row_length * i
        stop = start + row_length

        result = array.range(start, stop)
        StorageMethods.__equip(result, shape=(row_length,), dtype=array.dtype)
        return result

    @staticmethod
    # void(int64[:], int64, float64[:])
    def shuffle_global(idx, length, u01):
        raise NotImplementedError()

    @staticmethod
    # void(int64[:], float64[:], int64[:])
    def shuffle_local(idx, u01, cell_start):
        # TODO: print("Numba import!: ThrustRTC.shuffle_local(...)")

        from PySDM.backends.numba.numba import Numba
        host_idx = StorageMethods.to_ndarray(idx)
        host_u01 = StorageMethods.to_ndarray(u01)
        host_cell_start = StorageMethods.to_ndarray(cell_start)
        Numba.shuffle_local(host_idx, host_u01, host_cell_start)
        device_idx = StorageMethods.from_ndarray(host_idx)
        trtc.Copy(device_idx, idx)

    @staticmethod
    def to_ndarray(data):
        # TODO: move to __equip??
        if isinstance(data, StorageMethods.storage):
            pass
        elif isinstance(data, trtc.DVVector.DVRange):
            data_copy = StorageMethods.array(data.shape, float)
            trtc.Copy(data, data_copy)
            data = data_copy
        else:
            raise NotImplementedError()

        result = data.to_host()
        result = np.reshape(result, data.shape)
        return result

    @staticmethod
    def upload(numpy_data, backend_target):
        tmp = trtc.device_vector_from_numpy(numpy_data.flatten())
        trtc.Swap(tmp, backend_target)

    @staticmethod
    def write_row(array, i, row):
        row_length = array.shape[1]
        start = row_length * i
        stop = start + row_length

        trtc.Copy(row, array.range(start, stop))

    @staticmethod
    def __equip(data, shape, dtype):
        if isinstance(shape, int):
            shape = (shape,)
        data.shape = shape
        data.dtype = dtype
        data.get = lambda index: trtc.Reduce(data.range(index, index + 1))

