from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl.nice_thrust import nice_thrust

from ..conf import trtc


class PairMethods:
    __distance_pair_body = trtc.For(['data_out', 'data_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            data_out[(int64_t)(i/2)] = abs(data_in[i] - data_in[i + 1]);
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def distance_pair(data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        PairMethods.__distance_pair_body.launch_n(
            len(idx), [data_out.data, perm_in, is_first_in_pair.indicator.data])

    __find_pairs_body = trtc.For(['cell_start', 'perm_cell_id', 'is_first_in_pair', 'length'], "i", '''
        is_first_in_pair[i] = (
            i < length - 1 && // note: just to set the last element within the same loop
            perm_cell_id[i] == perm_cell_id[i+1] &&
            (i - cell_start[perm_cell_id[i]]) % 2 == 0
        );
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def find_pairs(cell_start, is_first_in_pair, cell_id, cell_idx, idx):  # TODO #330 handle cell_idx
        perm_cell_id = trtc.DVPermutation(cell_id.data, idx.data)
        d_length = trtc.DVInt64(len(idx))
        PairMethods.__find_pairs_body.launch_n(
            len(idx), [cell_start.data, perm_cell_id, is_first_in_pair.indicator.data, d_length])

    __max_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            data_out[(int64_t)(i/2)] = max(perm_in[i], perm_in[i + 1]);
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def max_pair(data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        PairMethods.__max_pair_body.launch_n(
            len(idx), [data_out.data, perm_in, is_first_in_pair.indicator.data])

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
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sort_pair(data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        if len(idx) > 1:
            PairMethods.__sort_pair_body.launch_n(
                len(idx) - 1, [data_out.data, perm_in, is_first_in_pair.indicator.data])

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
    def sort_within_pair_by_attr(idx, is_first_in_pair, attr):
        if len(idx) < 2:
            return
        PairMethods.__sort_within_pair_by_attr_body.launch_n(
            len(idx) - 1, [idx.data, is_first_in_pair.indicator.data, attr.data])

    __sum_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            data_out[(int64_t)(i/2)] = perm_in[i] + perm_in[i + 1];
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sum_pair(data_out, data_in, is_first_in_pair, idx):
        perm_in = trtc.DVPermutation(data_in.data, idx.data)
        trtc.Fill(data_out.data, trtc.DVDouble(0))
        PairMethods.__sum_pair_body.launch_n(
            len(idx), [data_out.data, perm_in, is_first_in_pair.indicator.data])
