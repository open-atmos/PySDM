"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc


class AlgorithmicStepMethods:

    @staticmethod
    def amax(row, idx, length):
        perm_in = trtc.DVPermutation(row, idx)
        index = trtc.Max_Element(perm_in.range(0, length))
        row_idx = idx.get(index)
        result = row.get(row_idx)
        return result

    @staticmethod
    def amin(row, idx, length):
        perm_in = trtc.DVPermutation(row, idx)
        index = trtc.Min_Element(perm_in.range(0, length))
        row_idx = idx.get(index)
        result = row.get(row_idx)
        return result

    @staticmethod
    def cell_id(cell_id, cell_origin, strides):
        loop = trtc.For(['cell_id', 'cell_origin', 'strides', 'n_dims', 'size'], "i", '''
                                    cell_id[i] = 0;
                                    for (int j = 0; j < n_dims; j++) 
                                    {
                                        cell_id[i] += cell_origin[size * i + j] * strides[j];
                                    }
                                ''')
        n_dims = trtc.DVInt64(strides.shape[1])
        size = trtc.DVInt64(cell_origin.shape[1])
        loop.launch_n(cell_id.size(), [cell_id, cell_origin, strides, n_dims, size])

    @staticmethod
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        loop = trtc.For(['data_out', 'data_in', 'is_first_in_pair', 'idx'], "i", '''
                                    if (is_first_in_pair[i]) 
                                    {
                                        data_out[i] = abs(data_in[idx[i]] - data_in[idx[i + 1]]);
                                    } else {
                                        data_out[i] = 0;
                                    }
                                ''')
        if length > 1:
            loop.launch_n(length - 1, [data_out, data_in, is_first_in_pair, idx])

    @staticmethod
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, sd_num):
        raise NotImplementedError()

    @staticmethod
    def max_pair(data_out, data_in, idx, length):
        perm_in = trtc.DVPermutation(data_in, idx)

        loop = trtc.For(['arr_in', 'arr_out'], "i", "arr_out[i] = max(arr_in[2 * i], arr_in[2 * i + 1]);")

        loop.launch_n(length // 2, [perm_in, data_out])

    @staticmethod
    def sum_pair(data_out, data_in, idx, length):
        perm_in = trtc.DVPermutation(data_in, idx)

        loop = trtc.For(['arr_in', 'arr_out'], "i", "arr_out[i] = arr_in[2 * i] + arr_in[2 * i + 1];")

        loop.launch_n(length // 2, [perm_in, data_out])
