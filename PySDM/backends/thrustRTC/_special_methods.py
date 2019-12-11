"""
Created at 10.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc
from ._maths_methods import MathsMethods


class SpecialMethods:

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
    def compute_gamma(prob, rand):
        MathsMethods.multiply(prob, -1.)
        loop = trtc.For(['prob', 'rand'], "i", "prob[i] += rand[int(i / 2)];")
        loop.launch_n(prob.size(), [prob, rand])
        MathsMethods.floor_in_place(prob)
        MathsMethods.multiply(prob, -1.)

    # TODO: add test, rethink...
    @staticmethod
    def first_element_is_zero(arr):
        return arr.get(0) == 0

    @staticmethod
    def cell_id(cell_id, cell_origin, strides):
        raise NotImplementedError()

    @staticmethod
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, sd_num):
        raise NotImplementedError()

    @staticmethod
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        raise NotImplementedError()

    @staticmethod
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        raise NotImplementedError()

    @staticmethod
    def normalize(prob, cell_id, cell_start, norm_factor, dt_div_dv):
        raise NotImplementedError()

    @staticmethod
    def apply_f_3_3(function, arg0, arg1, arg2, output0, output1, output2):
        raise NotImplementedError()

    @staticmethod
    def apply(function, args, output):
        raise NotImplementedError()






