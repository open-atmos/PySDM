"""
Created at 04.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64, boolean, prange
from PySDM.conf import NUMBA_PARALLEL


class SpecialMethods:
    @staticmethod
    @numba.njit(int64(int64[:], int64[:], int64))
    def remove_zeros(data, idx, length) -> int:
        new_length = 0
        for i in range(length):
            if data[idx[i]] == 0:
                idx[i] = len(idx)
            else:
                new_length += 1
        idx[:length].sort()
        return new_length

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64, float64[:, :], float64[:, :], float64[:], int64[:]),
                parallel=NUMBA_PARALLEL)
    def coalescence(n, idx, length, intensive, extensive, gamma, healthy):
        for i in prange(length - 1):
            if gamma[i] == 0:
                continue

            j = idx[i]
            k = idx[i + 1]

            if n[j] < n[k]:
                j, k = k, j
            g = min(gamma[i], n[j] // n[k])
            if g == 0:
                continue

            new_n = n[j] - g * n[k]
            if new_n > 0:
                n[j] = new_n
                extensive[:, k] += g * extensive[:, j]
            else:  # new_n == 0
                n[j] = n[k] // 2
                n[k] = n[k] - n[j]
                extensive[:, j] = g * extensive[:, j] + extensive[:, k]
                extensive[:, k] = extensive[:, j]
            if n[k] == 0 or n[j] == 0:
                healthy[0] = 0

    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), parallel=NUMBA_PARALLEL)
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = (data_in[idx[i]] + data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    @staticmethod
    @numba.njit(void(float64[:], int64[:], int64[:], int64[:], int64), parallel=NUMBA_PARALLEL)
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = max(data_in[idx[i]], data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    # TODO comment
    @staticmethod
    @numba.njit()
    def compute_gamma(prob, rand):
        prob *= -1.
        prob[0::2] += rand
        prob[1::2] += rand
        prob[:] = np.floor(prob)
        prob *= -1.

    @staticmethod
    @numba.njit(boolean(int64[:]))
    def first_element_is_zero(arr):
        return arr[0] == 0

    @staticmethod
    # @numba.njit() TODO: "np.dot() only supported on float and complex arrays"
    def cell_id(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin.T)

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64[:], int64[:], int64))
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, sd_num):
        for i in range(sd_num - 1, -1, -1):  # reversed
            cell_start[cell_id[idx[i]]] = i
        cell_start[-1] = sd_num

        for i in range(sd_num - 1):
            is_first_in_pair[i] = (
                cell_id[idx[i]] == cell_id[idx[i+1]] and
                (i - cell_start[cell_id[idx[i]]]) % 2 == 0
            )

    @staticmethod
    @numba.njit()
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        length = displacement.shape[0]
        for droplet in range(length):
            # Arakawa-C grid
            _l = (cell_origin[droplet, 0], cell_origin[droplet, 1])
            _r = (cell_origin[droplet, 0] + 1 * (dim == 0), cell_origin[droplet, 1] + 1 * (dim == 1))
            omega = position_in_cell[droplet, dim]
            displacement[droplet, dim] = scheme(omega, courant[_l], courant[_r])

    @staticmethod
    #@numba.njit()
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        moment_0[:] = 0
        moments[:, :] = 0
        for i in idx[:length]:
            if min_x < attr[x_id][i] < max_x:
                moment_0[cell_id[i]] += n[i]
                for k in range(specs_idx.shape[0]):
                    moments[k, cell_id[i]] += n[i] * attr[specs_idx[k], i] ** specs_rank[k]
        moments[:, :] /= moment_0  # TODO: should we divide or not...

    @staticmethod
    @numba.njit()
    def normalize(prob, cell_id, cell_start, norm_factor, dt_div_dv):
        n_cell = cell_start.shape[0]
        for i in range(n_cell - 1):
            sd_num = cell_start[i + 1] - cell_start[i]
            if sd_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = dt_div_dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
        for d in range(prob.shape[0]):
            prob[d] *= norm_factor[cell_id[d]]

    @staticmethod
    @numba.njit()
    def apply_f_3_3(function, arg0, arg1, arg2, output0, output1, output2):
        for i in range(output0.shape[0]):
            output0[i], output1[i], output2[i] = function(arg0[i], arg1[i], arg2[i])

    @staticmethod
    def apply(function, args, output):
        if len(args) == 3:
            if len(output) == 3:
                SpecialMethods.apply_f_3_3(function, *args, *output)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
