from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl import nice_thrust

from .precision_resolver import PrecisionResolver
from ..conf import trtc


class MomentsMethods:
    def __init__(self):

        self.__moments_body_0 = trtc.For(
            ['idx', 'min_x', 'attr_data', 'x_attr', 'max_x', 'moment_0', 'cell_id', 'n',
             'n_ranks', 'moments', 'ranks',
             'n_sd', 'n_cell'],
            "fake_i",
            '''
            auto i = idx[fake_i];
            if (min_x <= x_attr[i] && x_attr[i] < max_x) {
                atomicAdd((real_type*)&moment_0[cell_id[i]], (real_type)(n[i]));
                for (auto k = 0; k < n_ranks; k+=1) {
                    atomicAdd((real_type*) &moments[n_cell * k + cell_id[i]], n[i] * pow((real_type)(attr_data[i]), (real_type)(ranks[k])));
               }
            }
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__moments_body_1 = trtc.For(['n_ranks', 'moments', 'moment_0', 'n_cell'], "c_id", '''
            for (auto k = 0; k < n_ranks; k+=1) {
                if (moment_0[c_id] == 0) {
                    moments[n_cell * k  + c_id] = 0;
                } 
                else {
                    moments[n_cell * k + c_id] = moments[n_cell * k + c_id] / moment_0[c_id];
                }
            }
        ''')

        self.__spectrum_moments_body_0 = trtc.For(
            ['idx', 'attr_data', 'x_attr', 'moment_0', 'cell_id', 'n',
             'x_bins', 'n_bins', 'moments', 'rank', 'n_sd', 'n_cell'],
            "fake_i",
            '''
            auto i = idx[fake_i];
            for (auto k = 0; k < n_bins; k+=1) {
                if (x_bins[k] <= x_attr[i] and x_attr[i] < x_bins[k + 1]) {
                    atomicAdd((real_type*)&moment_0[n_cell * k + cell_id[i]], (real_type)(n[i]));
                    atomicAdd((real_type*) &moments[n_cell * k + cell_id[i]], n[i] * pow((real_type)(attr_data[i]), (real_type)(rank)));
                    break;
                }
            }
        '''.replace("real_type", PrecisionResolver.get_C_type()))

        self.__spectrum_moments_body_1 = trtc.For(['n_bins', 'moments', 'moment_0', 'n_cell'], "i", '''
            for (auto k = 0; k < n_bins; k+=1) {
                if (moment_0[n_cell * k + i] == 0) {
                    moments[n_cell * k  + i] = 0;
                } 
                else {
                    moments[n_cell * k + i] = moments[n_cell * k + i] / moment_0[n_cell * k + i];
                }
            }
        ''')

    @nice_thrust(**NICE_THRUST_FLAGS)
    def moments(self, moment_0, moments, n, attr_data, cell_id, idx, length, ranks,
                min_x, max_x, x_attr, weighting_attribute, weighting_rank):
        if weighting_rank != 0:
            raise NotImplementedError()

        n_cell = trtc.DVInt64(moments.shape[1])
        n_sd = trtc.DVInt64(moments.shape[0])
        n_ranks = trtc.DVInt64(ranks.shape[0])

        moments[:] = 0
        moment_0[:] = 0

        self.__moments_body_0.launch_n(length, [
            idx.data,
            PrecisionResolver.get_floating_point(min_x),
            attr_data.data,
            x_attr.data,
            PrecisionResolver.get_floating_point(max_x),
            moment_0.data,
            cell_id.data,
            n.data,
            n_ranks,
            moments.data,
            ranks.data,
            n_sd,
            n_cell
        ])

        self.__moments_body_1.launch_n(
            moment_0.shape[0], [n_ranks, moments.data, moment_0.data, n_cell])


    @nice_thrust(**NICE_THRUST_FLAGS)
    def spectrum_moments(self, moment_0, moments, n, attr_data, cell_id, idx, length,
            rank, x_bins, x_attr, weighting_attribute, weighting_rank):
        assert moments.shape[0] == x_bins.shape[0] - 1
        assert moment_0.shape == moments.shape
        if weighting_rank != 0:
            raise NotImplementedError()

        n_cell = trtc.DVInt64(moments.shape[1])
        n_sd = trtc.DVInt64(moments.shape[0])
        d_rank = trtc.DVInt64(rank)
        n_bins = trtc.DVInt64(len(x_bins) - 1)

        moments[:] = 0
        moment_0[:] = 0

        self.__spectrum_moments_body_0.launch_n(length, [
            idx.data,
            attr_data.data,
            x_attr.data,
            moment_0.data,
            cell_id.data,
            n.data,
            x_bins.data,
            n_bins,
            moments.data,
            d_rank,
            n_sd,
            n_cell
        ])

        self.__spectrum_moments_body_1.launch_n(
            moment_0.shape[1], [n_bins, moments.data, moment_0.data, n_cell])
