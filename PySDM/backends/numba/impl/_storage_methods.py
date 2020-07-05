"""
Created at 04.11.2019
"""

import numpy as np
import numba
from numba import void, float64, int64, prange
from PySDM.backends.numba import conf


class StorageMethods:
    integer = np.int64
    double = np.float64

    @staticmethod
    def array(shape, dtype):
        if dtype in (float, StorageMethods.double):
            data = np.full(shape, -1., dtype=StorageMethods.double)
        elif dtype in (int, StorageMethods.integer):
            data = np.full(shape, -1, dtype=StorageMethods.integer)
        else:
            raise NotImplementedError()
        return data

    @staticmethod
    def download(backend_data, numpy_target):
        np.copyto(numpy_target, backend_data, casting='safe')

    @staticmethod
    def from_ndarray(array):
        if str(array.dtype).startswith('int'):
            dtype = StorageMethods.integer
        elif str(array.dtype).startswith('float'):
            dtype = StorageMethods.double
        else:
            raise NotImplementedError()

        result = array.astype(dtype).copy()
        return result

    @staticmethod
    def range(array, start=0, stop=None):
        if stop is None:
            stop = array.shape[0]
        return array[start:stop]

    @staticmethod
    def read_row(array, i):
        return array[i, :]

    @staticmethod
    @numba.njit(void(int64[:], int64, float64[:]), **{**conf.JIT_FLAGS, **{'parallel': False}})
    def shuffle_global(idx, length, u01):
        for i in range(length-1, 0, -1):
            j = int(u01[i] * (i+1))
            idx[i], idx[j] = idx[j], idx[i]

    @staticmethod
    @numba.njit(void(int64[:], float64[:], int64[:]), **conf.JIT_FLAGS)
    def shuffle_local(idx, u01, cell_start):
        for c in prange(len(cell_start) - 1):
            for i in range(cell_start[c+1]-1, cell_start[c], -1):
                j = int(cell_start[c] + u01[i] * (cell_start[c+1] - cell_start[c]))
                idx[i], idx[j] = idx[j], idx[i]

    @staticmethod
    def to_ndarray(data):
        return data.copy()

    @staticmethod
    def upload(numpy_data, backend_target):
        np.copyto(backend_target.data, numpy_data, casting='safe')

    @staticmethod
    def write_row(array, i, row):
        array[i, :] = row

    @staticmethod
    def fill(array, value):
        array[:] = value
