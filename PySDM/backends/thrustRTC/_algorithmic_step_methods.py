"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc
from .nice_thrust import nice_thrust
from .conf import NICE_THRUST_FLAGS


class AlgorithmicStepMethods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def amax(row, idx, length):
        perm_in = trtc.DVPermutation(row, idx)
        index = trtc.Max_Element(perm_in.range(0, length))
        row_idx = idx.get(index)
        result = row.get(row_idx)
        return result

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def amin(row, idx, length):
        perm_in = trtc.DVPermutation(row, idx)
        index = trtc.Min_Element(perm_in.range(0, length))
        row_idx = idx.get(index)
        result = row.get(row_idx)
        return result

    __cell_id_body = trtc.For(['cell_id', 'cell_origin', 'strides', 'n_dims', 'size'], "i", '''
        cell_id[i] = 0;
        for (int j = 0; j < n_dims; j++) 
        {
            cell_id[i] += cell_origin[size * i + j] * strides[j];
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def cell_id(cell_id, cell_origin, strides):
        n_dims = trtc.DVInt64(strides.shape[1])
        size = trtc.DVInt64(cell_origin.shape[0])
        AlgorithmicStepMethods.__cell_id_body.launch_n(cell_id.size(), [cell_id, cell_origin, strides, n_dims, size])

    __distance_pair_body = trtc.For(['data_out', 'data_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) 
        {
            data_out[i] = abs(data_in[i] - data_in[i + 1]);
        } else {
            data_out[i] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        perm_in = trtc.DVPermutation(data_in, idx)
        if length > 1:
            AlgorithmicStepMethods.__distance_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])

    __find_pairs_body = trtc.For(['cell_start', 'perm_cell_id', 'is_first_in_pair'], "i", '''
        is_first_in_pair[i] = (
            perm_cell_id[i] == perm_cell_id[i+1] &&
            (i - cell_start[perm_cell_id[i]]) % 2 == 0
        );
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, length):
        perm_cell_id = trtc.DVPermutation(cell_id, idx)
        if length > 1:
            AlgorithmicStepMethods.__find_pairs_body.launch_n(length - 1, [cell_start, perm_cell_id, is_first_in_pair])

    __max_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) 
        {
            data_out[i] = max(perm_in[i], perm_in[i + 1]);
        } else {
            data_out[i] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        perm_in = trtc.DVPermutation(data_in, idx)
        if length > 1:
            AlgorithmicStepMethods.__max_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])

    __sum_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) 
        {
            data_out[i] = perm_in[i] + perm_in[i + 1];
        } else {
            data_out[i] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        perm_in = trtc.DVPermutation(data_in, idx)
        if length > 1:
            AlgorithmicStepMethods.__sum_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])
