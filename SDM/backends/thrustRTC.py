import ThrustRTC as trtc
import numpy as np


class ThrustRTC:
    storage = trtc.DVVector

    @staticmethod
    def array(shape, type):
        if type is float:
            elem_cls = 'double'
        elif type is int:
            elem_cls = 'int64_t'
        else:
            raise NotImplementedError

        data = trtc.device_vector(elem_cls, int(np.prod(shape)))
        # TODO: trtc.Fill(data, trtc.DVConstant(np.nan))

        ThrustRTC.__equip(data, shape)
        return data

    @staticmethod
    def __equip(data, shape):
        data.shape = shape
        data.get = lambda index: trtc.Reduce(data.range(index, index+1))

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

        ThrustRTC.__equip(result, shape)
        return result

    @staticmethod
    def to_ndarray(data: storage):
        result = data.to_host()
        result = np.reshape(result, data.shape)
        return result

    @staticmethod
    def shape(data):
        return data.shape

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
        pass  # idx = np.random.permutation(length)
        # Numpy.reindex(data, idx, length, axis=axis)

    @staticmethod
    def reindex(data, idx, length, axis):
        pass  # if axis == 1:
        #     data[:, 0:length] = data[:, idx]
        # else:
        #     raise NotImplementedError

    @staticmethod
    def argsort(idx, data, length):
        # TODO...
        copy = trtc.device_vector_from_dvs([data])
        trtc.Sort_By_Key(
            copy.range(0, length),
            idx.range(0, length)
        )

    @staticmethod
    def stable_argsort(idx: np.ndarray, data: np.ndarray, length: int):
        pass  # idx[0:length] = data[0:length].argsort(kind='stable')

    # @staticmethod
    # def item(data, index):
    #     result = trtc.Reduce(trtc.DVVector.DVRange(data, index, index+1))
    #     return result #data.to_host()[index]

    @staticmethod
    def amin(data):
        index = trtc.Min_Element(data)
        result = data.get(index) #ThrustRTC.item(data, index)
        return result

    @staticmethod
    def amax(data):
        index = trtc.Max_Element(data)
        result = data.get(index) # ThrustRTC.item(data, index)
        return result

    @staticmethod
    def urand(data, min=0, max=1):
        pass  # data[:] = np.random.uniform(min, max, data.shape)

    # TODO do not create array
    @staticmethod
    def remove_zeros(data, idx, length) -> int:
        pass  # for i in range(length):
        #     if data[0][idx[0][i]] == 0:
        #         idx[0][i] = idx.shape[1]
        # idx.sort()
        # return np.count_nonzero(data)

    @staticmethod
    def extensive_attr_coalescence(n, idx, length, data, gamma):
        pass  # # TODO in segments
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
        pass  # # TODO in segments
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
        perm_in = trtc.DVPermutation(data_in, idx)

        loop = trtc.For(['arr_in, arr_out'], "i",
                        '''
                        arr_out[i] = arr_in[2 * i] + arr_in[2 * i + 1]
                        ''')

        loop.launch_n(length // 2, [perm_in, data_out])

    @staticmethod
    def max_pair(data_out, data_in, idx, length):
        perm_in = trtc.DVPermutation(data_in, idx)

        loop = trtc.For(['arr_in, arr_out'], "i",
                        '''
                        arr_out[i] = max(arr_in[2 * i], arr_in[2 * i + 1])
                        ''')

        loop.launch_n(length // 2, [perm_in, data_out])

    @staticmethod
    def multiply(data, multiplier):
        loop = trtc.For(['arr', 'k'], "i", "arr[i] *= k;")
        const = trtc.DVDouble(multiplier)
        loop.launch_n(data.size(), [data, const])

    @staticmethod
    def sum(data_out, data_in):
        trtc.Transform_Binary(data_in, data_out, data_out, trtc.Plus)

    @staticmethod
    def floor(data):
        loop = trtc.For(['arr'], "i", "arr[i] = (long) arr[i];")
        loop.launch_n(data.size(), [data])
