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
        raise NotImplementedError()

    @staticmethod
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        raise NotImplementedError()

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
