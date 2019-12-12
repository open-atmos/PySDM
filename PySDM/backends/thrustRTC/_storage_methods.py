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

    @staticmethod
    def __equip(data, shape):
        data.shape = shape
        data.get = lambda index: trtc.Reduce(data.range(index, index + 1))

    @staticmethod
    def array(shape, dtype):
        if dtype is float:
            elem_cls = 'double'
        elif dtype is int:
            elem_cls = 'int64_t'
        else:
            raise NotImplementedError

        data = trtc.device_vector(elem_cls, int(np.prod(shape)))
        # TODO: trtc.Fill(data, trtc.DVConstant(np.nan))

        StorageMethods.__equip(data, shape)
        return data

    @staticmethod
    def download(backend_data, np_target):
        np_target[:] = backend_data.to_host()

    @staticmethod
    def dtype(data):
        elem_cls = data.name_elem_cls()
        if elem_cls == 'int64_t':
            nptype = np.int64
        elif elem_cls == 'double':
            nptype = np.float64
        else:
            raise NotImplemented()
        return nptype

    @staticmethod
    def from_ndarray(array):
        shape = array.shape

        if str(array.dtype).startswith('int'):
            dtype = np.int64
        elif str(array.dtype).startswith('float'):
            dtype = np.float64
        else:
            raise NotImplementedError

        if array.ndim > 1:
            array = array.astype(dtype).flatten()
        else:
            array = array.astype(dtype)

        result = trtc.device_vector_from_numpy(array)

        StorageMethods.__equip(result, shape)
        return result

    @staticmethod
    def read_row(array, i):
        row_length = array.shape[1]
        start = row_length * i
        stop = start + row_length

        result = array.range(start, stop)
        StorageMethods.__equip(result, shape=(row_length,))
        return result

    @staticmethod
    def shape(data):
        return data.shape

    @staticmethod
    def shuffle(data, length, axis):
        from PySDM.backends.numba.numba import Numba
        host_arr = StorageMethods.to_ndarray(data)
        Numba.shuffle(host_arr, length, axis=axis)
        dvce_arr = StorageMethods.from_ndarray(host_arr)
        trtc.Copy(dvce_arr, data)

        # TODO: take as argument (temporary memory)
        # idx = ThrustRTC.array((data.size(),), float)
        #
        # ThrustRTC.urand(idx)
        #
        # if axis == 0:
        #     trtc.Sort_By_Key(idx.range(0, length), data.range(0, length))
        # else:
        #     raise NotImplementedError

    @staticmethod
    def stable_argsort(idx, keys, length):
        raise NotImplementedError()

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
    def upload(np_data, backend_target):
        tmp = trtc.device_vector_from_numpy(np_data)
        trtc.Copy(tmp, backend_target)

    @staticmethod
    def write_row(array, i, row):
        row_length = array.shape[1]
        start = row_length * i
        stop = start + row_length

        trtc.Copy(row, array.range(start, stop))

