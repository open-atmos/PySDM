"""
Created at 10.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc
from ._maths_methods import MathsMethods


class AlgorithmicMethods:

    @staticmethod
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        raise NotImplementedError("function as argument => string?")

    @staticmethod
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy):
        loop = trtc.For(['n', 'volume', 'idx', 'intensive', 'extensive', 'gamma', 'healthy'], "i", '''
                        auto j = 2 * i;
                        auto k = j + 1;

                        j = idx[j];
                        k = idx[k];

                        if (n[j] < n[k]) {
                            auto old = j;
                            j = k;
                            k = old;
                        }

                        auto g = n[j] / n[k];
                        if (g > gamma[i])
                            g = gamma[i];

                        if (g != 0) {
                            auto new_n = n[j] - g * n[k];
                            if (new_n > 0) {
                                n[j] = new_n;
                                intensive[/*:*/, k] = (intensive[/*:*/, k] * volume[k] + intensive[/*:*/, j] * g * volume[j]) / (volume[k] + g * volume[j])
                                extensive[/*:,*/ k] += g * extensive[/*:,*/ j];
                            }
                            else {  // new_n == 0
                                n[j] = n[k] / 2;
                                n[k] = n[k] - n[j];
                                intensive[/*:*/, k] = (intensive[/*:*/, k] * volume[k] + intensive[/*:*/, j] * g * volume[j]) / (volume[k] + g * volume[j])
                                intensive[/*:*/, k] = intensive[/*:*/, j]
                                extensive[/*:,*/ j] = g * extensive[/*:,*/ j] + extensive[/*:,*/ k];
                                extensive[/*:,*/ k] = extensive[/*:,*/ j];
                            }
                            if (n[j] == 0 || n[k] == 0) {
                                healthy[0] = 0;
                            }
                        }
                            ''')
        loop.launch_n(length // 2, [n, volume, idx, intensive, extensive, gamma, healthy])

    @staticmethod
    def compute_gamma(prob, rand):
        MathsMethods.multiply(prob, -1.)
        loop = trtc.For(['prob', 'rand'], "i", "prob[i] += rand[int(i / 2)];")
        loop.launch_n(prob.size(), [prob, rand])
        MathsMethods.floor(prob)
        MathsMethods.multiply(prob, -1.)

    @staticmethod
    def condensation(
            solver,
            n_cell, cell_start_arg,
            v, particle_temperatures, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order
    ):
        raise NotImplementedError()

    @staticmethod
    def counting_sort_by_cell_id(new_idx, idx, cell_id, length, cell_start):
        raise NotImplementedError("Rethink")

    @staticmethod
    def counting_sort_by_cell_id_parallel(new_idx, idx, cell_id, length, cell_start, cell_start_p):
        raise NotImplementedError("Rethink")

    @staticmethod
    def flag_precipitated(cell_origin, position_in_cell, idx, length, healthy):
        idx_length = trtc.DVInt64(idx.size())
        loop = trtc.For(['idx', 'idx_length', 'n_dims', 'healthy', 'cell_origin', 'position_in_cell'], "i", '''
                        if (cell_origin[idx_length * (n_dims-1) + i] == 0 && position_in_cell[idx_length * (n_dims-1) + i] < 0) {
                            idx[i] = idx_length;
                            healthy[0] = 0;
                        }
                        ''')
        n_dims = len(cell_origin.shape)
        loop.launch_n(length, [idx, idx_length, n_dims, healthy, cell_origin, position_in_cell])

    @staticmethod
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        raise NotImplementedError()

    @staticmethod
    def normalize(prob, cell_id, cell_start, norm_factor, dt_div_dv):
        raise NotImplementedError()

    @staticmethod
    def remove_zeros(data, idx, length) -> int:
        idx_length = trtc.DVInt64(idx.size())

        # Warning (potential bug source): reading from outside of array
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
