"""
Created at 01.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from SDM.backends.numba import Numba
from SDM.conf import TRTC


if not TRTC:
    class ThrustRTC(Numba):
        pass
else:
    import ThrustRTC as trtc
    import CURandRTC as rndrtc


    class ThrustRTC:
        # TODO check static For
        storage = trtc.DVVector.DVVector

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
        def to_ndarray(data):
            # TODO: move to __equip??
            if isinstance(data, ThrustRTC.storage):
                pass
            elif isinstance(data, trtc.DVVector.DVRange):
                data_copy = ThrustRTC.array(data.shape, float)
                trtc.Copy(data, data_copy)
                data = data_copy
            else:
                raise NotImplementedError()

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
        def write_row(array, i, row):
            row_length = array.shape[1]
            start = row_length * i
            stop = start + row_length

            trtc.Copy(row, array.range(start, stop))

        @staticmethod
        def read_row(array, i):
            row_length = array.shape[1]
            start = row_length * i
            stop = start + row_length

            result = array.range(start, stop)
            ThrustRTC.__equip(result, shape=(row_length,))
            return result

        @staticmethod
        def shuffle(data, length, axis):
            from SDM.backends.numba import Numba
            host_arr = ThrustRTC.to_ndarray(data)
            Numba.shuffle(host_arr, length, axis=axis)
            dvce_arr = ThrustRTC.from_ndarray(host_arr)
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
        def amin(row, idx, length):
            perm_in = trtc.DVPermutation(row, idx)
            index = trtc.Min_Element(perm_in.range(0, length))
            row_idx = idx.get(index)
            result = row.get(row_idx)
            return result

        @staticmethod
        def amax(row, idx, length):
            perm_in = trtc.DVPermutation(row, idx)
            index = trtc.Max_Element(perm_in.range(0, length))
            row_idx = idx.get(index)
            result = row.get(row_idx)
            return result

        @staticmethod
        def urand(data):
            rng = rndrtc.DVRNG()
            # TODO: threads vs. blocks
            # TODO: proper state_init
            # TODO: generator choice
            chunks = min(32, data.size())  # TODO!!!
            ker = trtc.For(['rng', 'vec_rnd'], 'idx',
                           f'''
                           RNGState state;
                           rng.state_init(1234, idx, 0, state);  // initialize a state using the rng object
                           for (int i=0; i<{chunks}; i++)
                               vec_rnd[i+idx*{chunks}]=(float)state.rand01(); // generate random number using the rng object
                           ''')

            ker.launch_n(data.size()//chunks, [rng, data])

        @staticmethod
        def remove_zeros(data, idx, length) -> int:
            idx_length = trtc.DVInt64(idx.size())

            loop = trtc.For(['data', 'idx', 'idx_length'], "i", '''
                if (data[idx[i]] == 0)
                    idx[i] = idx_length;
                ''')
            loop.launch_n(length, [data, idx, idx_length])

            trtc.Sort(idx.range(0, length))

            result = trtc.Find(idx.range(0, length), idx_length)
            if result == -1:
                result = length

            return result

        @staticmethod
        def coalescence(n, idx, length, intensive, extensive, gamma, healthy):
            loop = trtc.For(['n', 'idx', 'data', 'gamma', 'healthy'], "i", '''
                auto j = 2 * i;
                auto k = j + 1;
    
                j = idx[j];
                k = idx[k];
    
                if (n[j] < n[k]) {
                    auto tmp = j;
                    j = k;
                    k = tmp;
                }
    
                auto g = n[j] / n[k];
                if (g > gamma[i])
                    g = gamma[i];
    
                if (g != 0) {
                    auto new_n = n[j] - g * n[k];
                    if (new_n > 0) {
                        n[j] = new_n;
                        data[/*:,*/ k] += g * data[/*:,*/ j];
                    }
                    else {  // new_n == 0
                        n[j] = n[k] / 2;
                        n[k] = n[k] - n[j];
                        data[/*:,*/ j] = g * data[/*:,*/ j] + data[/*:,*/ k];
                        data[/*:,*/ k] = data[/*:,*/ j];
                    }
                    if (n[j] == 0 || n[k] == 0) {
                        healthy[0] = 0;
                    }
                }
                    ''')
            loop.launch_n(length // 2, [n, idx, extensive, gamma, healthy])

        @staticmethod
        def sum_pair(data_out, data_in, idx, length):
            perm_in = trtc.DVPermutation(data_in, idx)

            loop = trtc.For(['arr_in', 'arr_out'], "i", "arr_out[i] = arr_in[2 * i] + arr_in[2 * i + 1];")

            loop.launch_n(length // 2, [perm_in, data_out])

        @staticmethod
        def max_pair(data_out, data_in, idx, length):
            perm_in = trtc.DVPermutation(data_in, idx)

            loop = trtc.For(['arr_in', 'arr_out'], "i", "arr_out[i] = max(arr_in[2 * i], arr_in[2 * i + 1]);")

            loop.launch_n(length // 2, [perm_in, data_out])

        @staticmethod
        def multiply(data, multiplier):
            if isinstance(multiplier, ThrustRTC.storage):
                loop = trtc.For(['arr', 'mult'], "i", "arr[i] *= mult[i];")
                mult = multiplier
            elif isinstance(multiplier, float):
                loop = trtc.For(['arr', 'mult'], "i", "arr[i] *= mult;")
                mult = trtc.DVDouble(multiplier)
            else:
                raise NotImplementedError()
            loop.launch_n(data.size(), [data, mult])

        @staticmethod
        def sum(data_out, data_in):
            trtc.Transform_Binary(data_in, data_out, data_out, trtc.Plus())

        @staticmethod
        def floor(data):
            loop = trtc.For(['arr'], "i", '''
                if (arr[i] >= 0) 
                    arr[i] = (long) arr[i];
                else
                {
                    auto tmp = arr[i];
                    arr[i] = (long) arr[i];
                    if (tmp != arr[i])
                        arr[i] -= 1;
                }
            ''')
            loop.launch_n(data.size(), [data])

        # TODO: add test, rethink...
        @staticmethod
        def first_element_is_zero(arr):
            return arr.get(0) == 0
