import ThrustRTC as trtc
import numpy as np


class ThrustRTC:
    storage = trtc.DVVector

    trtc.

    @staticmethod
    def array(shape, type):
        if type is float:
            elem_cls = 'double'
        elif type is int:
            elem_cls = 'int64_t'
        else:
            raise NotImplementedError

        size = shape[0]
        if len(shape) == 2:
            assert shape[0] == 1
            size = shape[1]

        data = trtc.device_vector(elem_cls, size)
        trtc.Fill(data, np.nan)

        return data

    @staticmethod
    def from_ndarray(array):
        if str(array.dtype).startswith('int'):
            dtype = np.int64
        elif str(array.dtype).startswith('float'):
            dtype = np.float64
        else:
            raise NotImplementedError

        if array.ndim > 1:
            array = array.astype(dtype)
        else:
            array = np.reshape(array.astype(dtype), (1, -1))

        result = trtc.device_vector_from_numpy(array)

        return result

    @staticmethod
    def to_ndarray(data: storage):
        return data.to_host().reshape(1, -1)

    @staticmethod
    def shape(data):
        return (1, data.size())

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
    def shuffle(data, length, axis):
        pass# idx = np.random.permutation(length)
        # Numpy.reindex(data, idx, length, axis=axis)

    @staticmethod
    def reindex(data, idx, length, axis):
        pass# if axis == 1:
        #     data[:, 0:length] = data[:, idx]
        # else:
        #     raise NotImplementedError

    @staticmethod
    def argsort(idx, data, length):
        pass# idx[0:length] = data[0:length].argsort()

    @staticmethod
    def stable_argsort(idx: np.ndarray, data: np.ndarray, length: int):
        pass# idx[0:length] = data[0:length].argsort(kind='stable')

    @staticmethod
    def amin(data):
        pass# result = np.amin(data)
        # return result

    @staticmethod
    def amax(data):
        pass# result = np.amax(data)
        # return result

    @staticmethod
    def transform(data, func, length):
        pass# data[:length] = np.fromfunction(
        #     np.vectorize(func, otypes=(data.dtype,)),
        #     (length,),
        #     dtype=np.int
        # )

    @staticmethod
    def foreach(data, func):
        pass# for i in range(len(data)):
        #     func(i)

    @staticmethod
    def urand(data, min=0, max=1):
        pass# data[:] = np.random.uniform(min, max, data.shape)

    # TODO do not create array
    @staticmethod
    def remove_zeros(data, idx, length) -> int:
        pass# for i in range(length):
        #     if data[0][idx[0][i]] == 0:
        #         idx[0][i] = idx.shape[1]
        # idx.sort()
        # return np.count_nonzero(data)

    @staticmethod
    def extensive_attr_coalescence(n, idx, length, data, gamma):
        pass# # TODO in segments
        # for i in range(length // 2):
        #     j = 2 * i
        #     k = j + 1
        #
        #     j = idx[j]
        #     k = idx[k]
        #
        #     if n[j] < n[k]:
        #         j, k = k, j
        #     g = min(gamma[i], n[j] // n[k])
        #
        #     new_n = n[j] - g * n[k]
        #     if new_n > 0:
        #         data[:, k] += g * data[:, j]
        #     else:  # new_n == 0
        #         data[:, j] = g * data[:, j] + data[:, k]
        #         data[:, k] = data[:, j]

    @staticmethod
    def n_coalescence(n, idx, length, gamma):
        pass# # TODO in segments
        # for i in range(length // 2):
        #     j = 2 * i
        #     k = j + 1
        #
        #     j = idx[j]
        #     k = idx[k]
        #
        #     if n[j] < n[k]:
        #         j, k = k, j
        #     g = min(gamma[i], n[j] // n[k])
        #
        #     new_n = n[j] - g * n[k]
        #     if new_n > 0:
        #         n[j] = new_n
        #     else:  # new_n == 0
        #         n[j] = n[k] // 2
        #         n[k] = n[k] - n[j]

    @staticmethod
    def sum_pair(data_out, data_in, idx, length):
        pass# for i in range(length // 2):
        #     data_out[i] = data_in[idx[2 * i]] + data_in[idx[2 * i + 1]]

    @staticmethod
    def max_pair(data_out, data_in, idx, length):
        pass# for i in range(length // 2):
        #     data_out[i] = max(data_in[idx[2 * i]], data_in[idx[2 * i + 1]])

    @staticmethod
    def multiply(data, multiplier):
        pass# data *= multiplier

    @staticmethod
    def sum(data_out, data_in):
        pass# data_out[:] = data_out + data_in

    @staticmethod
    def floor(data):
        pass

