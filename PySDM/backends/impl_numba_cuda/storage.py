"""
CPU Numpy-based implementation of Storage class
"""
import math
import numpy as np

from PySDM.backends.impl_common.storage_utils import (
    StorageBase,
    StorageSignature,
    empty,
    get_data_from_ndarray,
)
from PySDM.backends.impl_numba_cuda import storage_impl as impl
from numba import cuda


class Storage(StorageBase):
    FLOAT = np.float64
    INT = np.int64
    BOOL = np.bool_

    def __init__(self, signature):
        super().__init__(signature)

        self.threadsperblock = (128,)
        self.blockspergrid = (math.ceil(self.shape[0] / self.threadsperblock[0]),)

    def __getitem__(self, item):
        dim = len(self.shape)
        if isinstance(item, slice):
            step = item.step or 1
            if step != 1:
                raise NotImplementedError("step != 1")
            start = item.start or 0
            if dim == 1:
                stop = item.stop or len(self)
                result_data = self.data[item]
                result_shape = (stop - start,)
            elif dim == 2:
                stop = item.stop or self.shape[0]
                result_data = self.data[item]
                result_shape = (stop - start, self.shape[1])
            else:
                raise NotImplementedError(
                    "Only 2 or less dimensions array is supported."
                )
            if stop > self.data.shape[0]:
                raise IndexError(
                    f"requested a slice ({start}:{stop}) of Storage"
                    f" with first dim of length {self.data.shape[0]}"
                )
            result = Storage(StorageSignature(result_data, result_shape, self.dtype))
        elif isinstance(item, tuple) and dim == 2 and isinstance(item[1], slice):
            result = Storage(
                StorageSignature(self.data[item[0]], (*self.shape[1:],), self.dtype)
            )
        else:
            result = self.data[item]
        return result

    def __setitem__(self, key, value):
        if hasattr(value, "data"):
            self.data[key] = value.data
        else:
            self.data[key] = value
        return self

    def __iadd__(self, other):
        if hasattr(other, "data"):
            Storage.__inner_iadd_element_wise__[self.blockspergrid, self.threadsperblock](self.data, other.data)
        else:
            Storage.__inner_iadd_element__[self.blockspergrid, self.threadsperblock](self.data, other)

        return self

    @staticmethod
    @cuda.jit
    def __inner_iadd_element__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] += other

    @staticmethod
    @cuda.jit
    def __inner_iadd_element_wise__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] += other[pos]

    def __isub__(self, other):
        if hasattr(other, "data"):
            Storage.__inner_isub_element_wise__[self.blockspergrid, self.threadsperblock](self.data, other.data)
        else:
            Storage.__inner_isub_element__[self.blockspergrid, self.threadsperblock](self.data, other)

        return self

    @staticmethod
    @cuda.jit
    def __inner_isub_element__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] -= other

    @staticmethod
    @cuda.jit
    def __inner_isub_element_wise__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] -= other[pos]

    def __imul__(self, other):
        if hasattr(other, "data"):
            Storage.__inner_imul_element_wise__[self.blockspergrid, self.threadsperblock](self.data, other.data)
        else:
            Storage.__inner_imul_element__[self.blockspergrid, self.threadsperblock](self.data, other)

        return self

    @staticmethod
    @cuda.jit
    def __inner_imul_element__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] *= other

    @staticmethod
    @cuda.jit
    def __inner_imul_element_wise__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] *= other[pos]

    def __itruediv__(self, other):
        # TODO this should be changed to throw an exception if both arrays are integers (TypeError)
        if hasattr(other, "data"):
            Storage.__inner_itruediv_element_wise__[self.blockspergrid, self.threadsperblock](self.data, other.data)
        else:
            Storage.__inner_itruediv_element__[self.blockspergrid, self.threadsperblock](self.data, other)

        return self

    @staticmethod
    @cuda.jit
    def __inner_itruediv_element__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] /= other

    @staticmethod
    @cuda.jit
    def __inner_itruediv_element_wise__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] /= other[pos]

    def __imod__(self, other):
        if hasattr(other, "data"):
            Storage.__inner_imod_element_wise__[self.blockspergrid, self.threadsperblock](self.data, other.data)
        else:
            Storage.__inner_imod_element__[self.blockspergrid, self.threadsperblock](self.data, other)

        return self

    @staticmethod
    @cuda.jit
    def __inner_imod_element__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] %= other

    @staticmethod
    @cuda.jit
    def __inner_imod_element_wise__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] %= other[pos]

    def __ipow__(self, other):
        if hasattr(other, "data"):
            Storage.__inner_ipow_element_wise__[self.blockspergrid, self.threadsperblock](self.data, other.data)
        else:
            Storage.__inner_ipow_element__[self.blockspergrid, self.threadsperblock](self.data, other)

        return self

    @staticmethod
    @cuda.jit
    def __inner_ipow_element__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] = pow(output[pos], other)

    @staticmethod
    @cuda.jit
    def __inner_ipow_element_wise__(output, other):
        tx = cuda.threadIdx.x
        bw = cuda.blockDim.x
        bi = cuda.blockIdx.x
        pos = tx + bw * bi

        if pos < len(output):  # Check array boundaries
            output[pos] = pow(output[pos], other[pos])

    def __bool__(self):
        if len(self) == 1:
            result = bool(self.data[0] != 0)
        else:
            raise NotImplementedError("Logic value of array is ambiguous.")
        return result

    def detach(self):
        if self.data.base is not None:
            self.data = np.array(self.data)

    def download(self, target, reshape=False):
        if reshape:
            data = self.data.reshape(target.shape)
        else:
            data = self.data
        np.copyto(target, data, casting="safe")

    @staticmethod
    def _get_empty_data(shape, dtype):
        if dtype in (float, Storage.FLOAT):
            data = cuda.device_array(shape, dtype=Storage.FLOAT)
            dtype = Storage.FLOAT
        elif dtype in (int, Storage.INT):
            data = cuda.device_array(shape, dtype=Storage.INT)
            dtype = Storage.INT
        elif dtype in (bool, Storage.BOOL):
            data = cuda.device_array(shape, dtype=Storage.BOOL)
            dtype = Storage.BOOL
        else:
            raise NotImplementedError()

        return StorageSignature(data, shape, dtype)

    @staticmethod
    def empty(shape, dtype):
        return empty(shape, dtype, Storage)

    @staticmethod
    def _get_data_from_ndarray(array):
        return get_data_from_ndarray(
            array=array,
            storage_class=Storage,
            copy_fun=lambda array_astype: cuda.to_device(
                array_astype.ravel()
            ),
        )

    def amin(self):
        return impl.amin(self.data)

    def all(self):
        return self.data.all()

    @staticmethod
    def from_ndarray(array):
        result = Storage(Storage._get_data_from_ndarray(array))
        return result

    def floor(self, other=None):
        if other is None:
            impl.floor(self.data)
        else:
            impl.floor_out_of_place(self.data, other.data)
        return self

    def product(self, multiplicand, multiplier):
        if hasattr(multiplier, "data"):
            impl.multiply_out_of_place(self.data, multiplicand.data, multiplier.data)
        else:
            impl.multiply_out_of_place(self.data, multiplicand.data, multiplier)
        return self

    def ratio(self, dividend, divisor):
        impl.divide_out_of_place(self.data, dividend.data, divisor.data)
        return self

    def divide_if_not_zero(self, divisor):
        impl.divide_if_not_zero(self.data, divisor.data)
        return self

    def sum(self, arg_a, arg_b):
        impl.sum_out_of_place(self.data, arg_a.data, arg_b.data)
        return self

    def ravel(self, other):
        if isinstance(other, Storage):
            self.data[:] = other.data.ravel()
        else:
            self.data[:] = other.ravel()

    def urand(self, generator):
        generator(self)

    def to_ndarray(self):
        res = self.data.copy_to_host()
        return res

    def upload(self, data):
        # cuda.to_device(data, to=self.data)
        self.data = cuda.to_device(data)

    def fill(self, other):
        if isinstance(other, Storage):
            self.data[:] = other.data
        else:
            self.data[:] = other
