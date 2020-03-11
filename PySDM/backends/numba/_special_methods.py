"""
Created at 04.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import numba
from numba import void, float64, int64, boolean, prange
from PySDM.backends.numba import conf


class SpecialMethods:

    @staticmethod
    @numba.njit(int64(int64[:], int64[:], int64), **{**conf.JIT_FLAGS, **{'parallel': False}})
    def remove_zeros(data, idx, length) -> int:
        new_length = 0
        for i in range(length):
            if idx[i] == len(idx) or data[idx[i]] == 0:
                idx[i] = len(idx)
            else:
                new_length += 1
        idx[:length].sort()
        return new_length

    @staticmethod
    @numba.njit(void(int64[:, :], float64[:, :], int64[:], int64, int64[:]))
    def flag_precipitated(cell_origin, position_in_cell, idx, length, healthy):
        for i in range(length):
            if cell_origin[i, -1] == 0 and position_in_cell[i, -1] < 0:
                idx[i] = len(idx)
                healthy[0] = 0

    @staticmethod
    @numba.njit(void(int64[:], float64[:], int64[:], int64, float64[:, :], float64[:, :], float64[:], int64[:]),
                **{**conf.JIT_FLAGS, **{'parallel': False}}) # TODO: waits for https://github.com/numba/numba/issues/5279
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy):
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

            # note: extensive must be modified after intensive (as it is used as weights)
            new_n = n[j] - g * n[k]
            if new_n > 0:
                n[j] = new_n
                intensive[:, k] = (intensive[:, k] * volume[k] + intensive[:, j] * g * volume[j]) / (volume[k] + g * volume[j])
                extensive[:, k] += g * extensive[:, j]
            else:  # new_n == 0
                n[j] = n[k] // 2
                n[k] = n[k] - n[j]
                intensive[:, j] = (intensive[:, k] * volume[k] + intensive[:, j] * g * volume[j]) / (volume[k] + g * volume[j])
                intensive[:, k] = intensive[:, j]
                extensive[:, j] = g * extensive[:, j] + extensive[:, k]
                extensive[:, k] = extensive[:, j]
            if n[k] == 0 or n[j] == 0:
                healthy[0] = 0

    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = (data_in[idx[i]] + data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    @staticmethod
    @numba.njit(void(float64[:], float64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = np.abs(data_in[idx[i]] - data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    @staticmethod
    @numba.njit(void(float64[:], int64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        for i in prange(length - 1):
            data_out[i] = max(data_in[idx[i]], data_in[idx[i + 1]]) if is_first_in_pair[i] else 0

    # TODO comment
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def compute_gamma(prob, rand):
        for i in prange(len(prob)):
            prob[i] *= -1.
            prob[i] += rand[i//2]
            prob[i] = -np.floor(prob[i])

    @staticmethod
    @numba.njit(boolean(int64[:]), **{**conf.JIT_FLAGS, **{'parallel': False}})
    def first_element_is_zero(arr):
        return arr[0] == 0

    @staticmethod
    #@numba.njit(**conf.JIT_FLAGS) # TODO: "np.dot() only supported on float and complex arrays"
    def cell_id(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin.T)

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64[:], int64[:], int64), **conf.JIT_FLAGS)
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, sd_num):
        for i in prange(sd_num - 1):
            is_first_in_pair[i] = (
                cell_id[idx[i]] == cell_id[idx[i+1]] and
                (i - cell_start[cell_id[idx[i]]]) % 2 == 0
            )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        length = displacement.shape[0]
        for droplet in prange(length):
            # Arakawa-C grid
            _l = (cell_origin[droplet, 0], cell_origin[droplet, 1])
            _r = (cell_origin[droplet, 0] + 1 * (dim == 0), cell_origin[droplet, 1] + 1 * (dim == 1))
            omega = position_in_cell[droplet, dim]
            displacement[droplet, dim] = scheme(omega, courant[_l], courant[_r])

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        moment_0[:] = 0
        moments[:, :] = 0
        for i in idx[:length]:
            if min_x < attr[x_id][i] < max_x:
                moment_0[cell_id[i]] += n[i]
                for k in range(specs_idx.shape[0]):
                    moments[k, cell_id[i]] += n[i] * attr[specs_idx[k], i] ** specs_rank[k]
        moments[:, :] /= moment_0

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def normalize(prob, cell_id, cell_start, norm_factor, dt_div_dv):
        n_cell = cell_start.shape[0]  # TODO: isn't it n_cell_plus_one?
        for i in range(n_cell - 1):
            sd_num = cell_start[i + 1] - cell_start[i]
            if sd_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = dt_div_dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
        for d in range(prob.shape[0]):
            prob[d] *= norm_factor[cell_id[d]]

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
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

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64[:], int64, int64[:]), **conf.JIT_FLAGS)
    def countsort_by_cell_id(new_idx, idx, cell_id, length, cell_start):
        cell_end = cell_start

        cell_end[:] = 0
        for i in range(length):
            cell_end[cell_id[idx[i]]] += 1
        for i in range(1, len(cell_end)):  # TODO: if len(cell_end) != n_cell+1 silently does wrong thing...
            cell_end[i] += cell_end[i - 1]
        for i in range(length-1, -1, -1):
            cell_end[cell_id[idx[i]]] -= 1
            new_idx[cell_end[cell_id[idx[i]]]] = idx[i]

    @staticmethod
    @numba.njit(void(int64[:], int64[:], int64[:], int64, int64[:], int64[:, :]), parallel=True)
    def countsort_by_cell_id_parallel(new_idx, idx, cell_id, length, cell_start, cell_start_p):
        cell_end_thread = cell_start_p

        thread_num = cell_end_thread.shape[0]
        for t in prange(thread_num):
            cell_end_thread[t, :] = 0
            for i in range(t * length // thread_num,
                           (t + 1) * length // thread_num if t < thread_num - 1 else length):
                cell_end_thread[t, cell_id[idx[i]]] += 1

        cell_start[:] = np.sum(cell_end_thread, axis=0)
        for i in range(1, len(cell_start)):  # TODO: if len(cell_end) != n_cell+1 silently does wrong thing...
            cell_start[i] += cell_start[i - 1]

        tmp = cell_end_thread[0, :]
        tmp[:] = cell_end_thread[thread_num - 1, :]
        cell_end_thread[thread_num - 1, :] = cell_start[:]
        for t in range(thread_num - 2, -1, -1):
            cell_start[:] = cell_end_thread[t + 1, :] - tmp[:]
            tmp[:] = cell_end_thread[t, :]
            cell_end_thread[t, :] = cell_start[:]

        for t in prange(thread_num):
            for i in range((t + 1) * length // thread_num - 1 if t < thread_num - 1 else length - 1,
                           t * length // thread_num - 1,
                           -1):
                cell_end_thread[t, cell_id[idx[i]]] -= 1
                new_idx[cell_end_thread[t, cell_id[idx[i]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]

    @staticmethod
    def condensation(
            solver,
            n_cell, cell_start_arg,
            v, particle_temperatures, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order
    ):
        n_threads = min(numba.config.NUMBA_NUM_THREADS, n_cell)
        SpecialMethods._condensation(
            solver, n_threads, n_cell, cell_start_arg,
            v, particle_temperatures, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def _condensation(
            solver, n_threads, n_cell, cell_start_arg,
            v, particle_temperatures, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order
    ):
        for thread_id in numba.prange(n_threads):
            for i in range(thread_id, n_cell, n_threads):  # TODO: at least show that it is not slower :)
                cell_id = cell_order[i]

                cell_start = cell_start_arg[cell_id]
                cell_end = cell_start_arg[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                dthd_dt = (pthd[cell_id] - thd[cell_id]) / dt
                dqv_dt = (pqv[cell_id] - qv[cell_id]) / dt
                md_new = prhod[cell_id] * dv
                md_old = rhod[cell_id] * dv
                md_mean = (md_new + md_old) / 2
                rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2

                qv_new, thd_new, substeps_hint = solver(
                    v, particle_temperatures, n, vdry,
                    idx[cell_start:cell_end],  # TODO
                    kappa, thd[cell_id], qv[cell_id], dthd_dt, dqv_dt, md_mean, rhod_mean,
                    rtol_x, rtol_thd, dt, substeps[cell_id]
                )

                substeps[cell_id] = substeps_hint

                pqv[cell_id] = qv_new
                pthd[cell_id] = thd_new
