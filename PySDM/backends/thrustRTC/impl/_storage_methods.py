"""
Created at 10.12.2019
"""

from ..conf import trtc
import numpy as np
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS


class StorageMethods:
    storage = trtc.DVVector.DVVector
    integer = np.int64
    double = np.float64

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
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

        StorageMethods.__equip(data, shape, elem_dtype)
        return data

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def download(backend_data, numpy_target):
        if isinstance(backend_data, StorageMethods.storage):
            numpy_target[:] = np.reshape(backend_data.to_host(), backend_data.shape)
        else:
            numpy_target[:] = StorageMethods.to_ndarray(backend_data)

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
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
    @nice_thrust(**NICE_THRUST_FLAGS)
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
            raise NotImplementedError("Only 2 or less dimensions array is supported.")
        StorageMethods.__equip(result, shape=new_shape, dtype=array.dtype)
        return result

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def read_row(array, i):
        row_length = array.shape[1]
        start = row_length * i
        stop = start + row_length

        result = array.range(start, stop)
        StorageMethods.__equip(result, shape=(row_length,), dtype=array.dtype)
        return result

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def shuffle_global(idx, length, u01):
        # WARNING: ineffective implementation
        raise NotImplementedError()
        # trtc.Sort_By_Key(u01.range(0, length), idx.range(0, length))

    __shuffle_local_body = trtc.For(['cell_start', 'u01', 'idx'], "i", '''
        for (int k = cell_start[i+1]-1; k > cell_start[i]; k -= 1) {
            int j = cell_start[i] + (int)( u01[k] * (cell_start[i+1] - cell_start[i]) );
            int tmp = idx[k];
            idx[k] = idx[j];
            idx[j] = tmp;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def shuffle_local(idx, u01, cell_start):
        StorageMethods.__shuffle_local_body.launch_n(cell_start.size() - 1, [cell_start, u01, idx])

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def to_ndarray(data):
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
    @nice_thrust(**NICE_THRUST_FLAGS)
    def upload(numpy_data, backend_target):
        tmp = trtc.device_vector_from_numpy(numpy_data.flatten())
        trtc.Swap(tmp, backend_target)

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
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

        def get(index):
            return trtc.Reduce(data.range(index, index + 1))

        data.get = get

