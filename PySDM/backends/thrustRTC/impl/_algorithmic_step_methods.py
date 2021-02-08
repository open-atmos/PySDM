"""
Created at 20.03.2020
"""

from ..conf import trtc
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS


class AlgorithmicStepMethods:

    __cell_id_body = trtc.For(['cell_id', 'cell_origin', 'strides', 'n_dims', 'size'], "i", '''
        cell_id[i] = 0;
        for (auto j = 0; j < n_dims; j += 1) {
            cell_id[i] += cell_origin[size * j + i] * strides[j];
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def cell_id(cell_id, cell_origin, strides):
        assert cell_origin.shape[0] == strides.shape[1]
        assert cell_id.shape[0] == cell_origin.shape[1]
        assert strides.shape[0] == 1
        n_dims = trtc.DVInt64(cell_origin.shape[0])
        size = trtc.DVInt64(cell_origin.shape[1])
        AlgorithmicStepMethods.__cell_id_body.launch_n(len(cell_id), [cell_id.data, cell_origin.data, strides.data, n_dims, size])

    __distance_pair_body = trtc.For(['data_out', 'data_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            data_out[(int64_t)(i/2)] = abs(data_in[i] - data_in[i + 1]);
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        perm_in = trtc.DVPermutation(data_in, idx)
        trtc.Fill(data_out, trtc.DVDouble(0))
        AlgorithmicStepMethods.__distance_pair_body.launch_n(length, [data_out, perm_in, is_first_in_pair])

    __find_pairs_body = trtc.For(['cell_start', 'perm_cell_id', 'is_first_in_pair', 'length'], "i", '''
        is_first_in_pair[i] = (
            i < length - 1 &&
            perm_cell_id[i] == perm_cell_id[i+1] &&
            (i - cell_start[perm_cell_id[i]]) % 2 == 0
        );
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def find_pairs(cell_start, is_first_in_pair, cell_id, cell_idx, idx):  # TODO #330 handle cell_idx
        perm_cell_id = trtc.DVPermutation(cell_id.data, idx.data)
        d_length = trtc.DVInt64(len(idx))  # TODO #350 length-1 as we use i+1 in __find_pairs_body
        AlgorithmicStepMethods.__find_pairs_body.launch_n(
            len(idx), [cell_start.data, perm_cell_id, is_first_in_pair.indicator.data, d_length])

    __max_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            data_out[(int64_t)(i/2)] = max(perm_in[i], perm_in[i + 1]);
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        perm_in = trtc.DVPermutation(data_in, idx)
        trtc.Fill(data_out, trtc.DVDouble(0))
        AlgorithmicStepMethods.__max_pair_body.launch_n(length, [data_out, perm_in, is_first_in_pair])

    __sort_pair_body = trtc.For(['data_out', 'data_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            if (data_in[i] < data_in[i + 1]) {
                data_out[i] = data_in[i + 1];
                data_out[i + 1] = data_in[i];
            } 
            else {
                data_out[i] = data_in[i];
                data_out[i + 1] = data_in[i + 1];
            }
        } 
        else {
            data_out[i] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sort_pair(data_out, data_in, is_first_in_pair, idx, length):
        perm_in = trtc.DVPermutation(data_in, idx)
        trtc.Fill(data_out, trtc.DVDouble(0))
        if length > 1:
            AlgorithmicStepMethods.__sort_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])

    __sort_within_pair_by_attr_body = trtc.For(["idx", "is_first_in_pair", "attr"], "i", '''
        if (is_first_in_pair[i]) {
            if (attr[idx[i]] < attr[idx[i + 1]]) {
                auto tmp = idx[i];
                idx[i] = idx[i + 1];
                idx[i + 1] = tmp;
            }
        }
        ''')

    @staticmethod
    def sort_within_pair_by_attr(idx, length, is_first_in_pair, attr):
        AlgorithmicStepMethods.__sort_within_pair_by_attr_body.launch_n(
            length - 1, [idx.data, is_first_in_pair.indicator.data, attr.data])

    __sum_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            data_out[(int64_t)(i/2)] = perm_in[i] + perm_in[i + 1];
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        perm_in = trtc.DVPermutation(data_in, idx)
        trtc.Fill(data_out, trtc.DVDouble(0))
        AlgorithmicStepMethods.__sum_pair_body.launch_n(length, [data_out, perm_in, is_first_in_pair])
