"""
Created at 04.11.2019
"""

import numba
import numpy as np
from numba import void, f8, i8, prange, b1

from PySDM.backends.numba import conf
from PySDM.backends.numba.storage import Storage


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def pair_indices(i, idx, is_first_in_pair):
    offset = 1 - is_first_in_pair[2 * i]
    j = idx[2 * i + offset]
    k = idx[2 * i + 1 + offset]
    return j, k


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def calculate_displacement_body_common(dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell):
    omega = position_in_cell[dim, droplet]
    displacement[dim, droplet] = scheme(omega, courant[_l], courant[_r])


class AlgorithmicMethods:
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def adaptive_sdm_end_body(dt_left, n_cell, cell_start):
        end = 0
        for i in range(n_cell - 1, -1, -1):
            if dt_left[i] == 0:
                continue
            else:
                end = cell_start[i + 1]
                break
        return end

    @staticmethod
    def adaptive_sdm_end(dt_left, cell_start):
        return AlgorithmicMethods.adaptive_sdm_end_body(dt_left.data, len(dt_left), cell_start.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def adaptive_sdm_gamma_body(gamma, idx, length, n, cell_id, dt_left, dt, dt_range, is_first_in_pair,
                                stats_n_substep, stats_dt_min):
        dt_todo = np.empty_like(dt_left)
        for cid in prange(len(dt_todo)):
            dt_todo[cid] = min(dt_left[cid], dt_range[1])
        for i in range(length // 2):  # TODO #401
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = n[j] // n[k]
            dt_optimal = dt * prop / gamma[i]
            cid = cell_id[j]
            dt_optimal = max(dt_optimal, dt_range[0])
            dt_todo[cid] = min(dt_todo[cid], dt_optimal)
            stats_dt_min[cid] = min(stats_dt_min[cid], dt_optimal)
        for i in prange(length // 2):
            if gamma[i] == 0:
                continue
            j, _ = pair_indices(i, idx, is_first_in_pair)
            gamma[i] *= dt_todo[cell_id[j]] / dt
        for cid in prange(len(dt_todo)):
            dt_left[cid] -= dt_todo[cid]
            if dt_todo[cid] > 0:
                stats_n_substep[cid] += 1

    @staticmethod
    def adaptive_sdm_gamma(gamma, n, cell_id, dt_left, dt, dt_range, is_first_in_pair, stats_n_substep, stats_dt_min):
        return AlgorithmicMethods.adaptive_sdm_gamma_body(
            gamma.data, n.idx.data, len(n), n.data, cell_id.data,
            dt_left.data, dt, dt_range, is_first_in_pair.indicator.data, stats_n_substep.data, stats_dt_min.data)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
    def calculate_displacement_body_1d(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        length = displacement.shape[1]
        for droplet in prange(length):
            # Arakawa-C grid
            _l = cell_origin[0, droplet]
            _r = cell_origin[0, droplet] + 1
            calculate_displacement_body_common(dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
    def calculate_displacement_body_2d(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        length = displacement.shape[1]
        for droplet in prange(length):
            # Arakawa-C grid
            _l = (cell_origin[0, droplet], cell_origin[1, droplet])
            _r = (cell_origin[0, droplet] + 1 * (dim == 0), cell_origin[1, droplet] + 1 * (dim == 1))
            calculate_displacement_body_common(dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell)

    @staticmethod
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        n_dims = len(courant.shape)
        if n_dims == 1:
            AlgorithmicMethods.calculate_displacement_body_1d(dim, scheme, displacement.data, courant.data, cell_origin.data,
                                                              position_in_cell.data)
        elif n_dims == 2:
            AlgorithmicMethods.calculate_displacement_body_2d(dim, scheme, displacement.data, courant.data, cell_origin.data,
                                                              position_in_cell.data)
        else:
            raise NotImplementedError()

    @staticmethod
    # @numba.njit(**conf.JIT_FLAGS)  # Note: in Numba 0.51 "np.dot() only supported on float and complex arrays"
    def cell_id_body(cell_id, cell_origin, strides):
        if strides.size == 1:
            cell_id[:] = cell_origin
        else:
            cell_id[:] = np.dot(strides, cell_origin)

    @staticmethod
    def cell_id(cell_id, cell_origin, strides):
        return AlgorithmicMethods.cell_id_body(cell_id.data, cell_origin.data, strides.data)

    @staticmethod
    @numba.njit(
        void(i8[:], f8[:], i8[:], i8, f8[:, :], f8[:, :], f8[:], i8[:], b1[:]),
        **conf.JIT_FLAGS)
    def coalescence_body(n, volume, idx, length, intensive, extensive, gamma, healthy, is_first_in_pair):
        for i in prange(length // 2):
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)

            new_n = n[j] - gamma[i] * n[k]
            if new_n > 0:
                n[j] = new_n
                for ii in range(0, len(intensive)):
                    intensive[ii, k] = (intensive[ii, k] * volume[k] + intensive[ii, j] * gamma[i] * volume[j]) \
                                      / (volume[k] + gamma[i] * volume[j])
                for ie in range(0, len(extensive)):
                    extensive[ie, k] += gamma[i] * extensive[ie, j]
            else:  # new_n == 0
                n[j] = n[k] // 2
                n[k] = n[k] - n[j]
                for ii in range(0, len(intensive)):
                    intensive[ii, j] = (intensive[ii, k] * volume[k] + intensive[ii, j] * gamma[i] * volume[j]) \
                                      / (volume[k] + gamma[i] * volume[j])
                    intensive[ii, k] = intensive[ii, j]
                for ie in range(0, len(extensive)):
                    extensive[ie, j] = gamma[i] * extensive[ie, j] + extensive[ie, k]
                    extensive[ie, k] = extensive[ie, j]
            if n[k] == 0 or n[j] == 0:
                healthy[0] = 0

    @staticmethod
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy, is_first_in_pair):
        AlgorithmicMethods.coalescence_body(n.data, volume.data, idx.data, length, intensive.data,
                                            extensive.data, gamma.data, healthy.data, is_first_in_pair.indicator.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def compute_gamma_body(gamma, rand, idx, length, n, cell_id,
                           collision_rate_deficit, collision_rate, is_first_in_pair):

        """
        return in "gamma" array gamma (see: http://doi.org/10.1002/qj.441, section 5)
        formula:
        gamma = floor(prob) + 1 if rand <  prob - floor(prob)
              = floor(prob)     if rand >= prob - floor(prob)
        """
        for i in prange(length // 2):
            gamma[i] = np.ceil(gamma[i] - rand[i])

            if gamma[i] == 0:
                continue

            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = n[j] // n[k]
            g = min(int(gamma[i]), prop)  # TODO #416: test asserting that min is not needed with adaptivity
            cid = cell_id[j]
            collision_rate[cid] += g * n[k]
            collision_rate_deficit[cid] += (int(gamma[i]) - g) * n[k]
            gamma[i] = g

    @staticmethod
    def compute_gamma(gamma, rand, n, cell_id,
                      collision_rate_deficit, collision_rate, is_first_in_pair):
        return AlgorithmicMethods.compute_gamma_body(
            gamma.data, rand.data, n.idx.data, len(n), n.data, cell_id.data,
            collision_rate_deficit.data, collision_rate.data, is_first_in_pair.indicator.data)

    @staticmethod
    def condensation(
            solver,
            n_cell, cell_start_arg,
            v, particle_temperatures, r_cr, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order, ripening_flags, RH_max
    ):
        n_threads = min(numba.get_num_threads(), n_cell)
        AlgorithmicMethods._condensation(
            solver, n_threads, n_cell, cell_start_arg.data,
            v.data, particle_temperatures.data, r_cr.data, n.data, vdry.data, idx.data,
            rhod.data, thd.data, qv.data, dv, prhod.data, pthd.data, pqv.data, kappa,
            rtol_x, rtol_thd, dt, substeps.data, cell_order, ripening_flags.data, RH_max.data
        )

    @staticmethod
    @numba.njit(f8(i8[:, :], f8[:, :], f8[:], i8[:], i8[:], i8, i8[:]))
    def flag_precipitated_body(cell_origin, position_in_cell, volume, n, idx, length, healthy):
        rainfall = 0.
        flag = len(idx)
        for i in range(length):
            if cell_origin[-1, idx[i]] + position_in_cell[-1, idx[i]] < 0:
                rainfall += volume[idx[i]] * n[idx[i]]
                idx[i] = flag
                healthy[0] = 0
        return rainfall

    @staticmethod
    def flag_precipitated(cell_origin, position_in_cell, volume, n, idx, length, healthy) -> float:
        return AlgorithmicMethods.flag_precipitated_body(
            cell_origin.data, position_in_cell.data, volume.data, n.data, idx.data, length, healthy.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def linear_collection_efficiency_body(params, output, radii, is_first_in_pair, idx, length, unit):
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        output[:] = 0
        for i in prange(length - 1):
            if is_first_in_pair[i]:
                if radii[idx[i]] > radii[idx[i + 1]]:
                    r = radii[idx[i]] / unit
                    r_s = radii[idx[i + 1]] / unit
                else:
                    r = radii[idx[i + 1]] / unit
                    r_s = radii[idx[i]] / unit
                p = r_s / r
                if p != 0 and p != 1:
                    G = (G1 / r) ** Mg + G2 + G3 * r
                    Gp = (1 - p) ** G
                    if Gp != 0:
                        D = D1 / r ** D2
                        E = E1 / r ** E2
                        F = (F1 / r) ** Mf + F2
                        output[i // 2] = A + B * p + D / p ** F + E / Gp
                        output[i // 2] = max(0, output[i // 2])

    @staticmethod
    def linear_collection_efficiency(params, output, radii, is_first_in_pair, unit):
        return AlgorithmicMethods.linear_collection_efficiency_body(
            params, output.data, radii.data, is_first_in_pair.indicator.data, radii.idx.data, len(is_first_in_pair), unit)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def interpolation_body(output, radius, factor, b, c):
        for i in prange(len(radius)):
            r_id = int(factor * radius[i])
            r_rest = ((factor * radius[i]) % 1) / factor
            output[i] = b[r_id] + r_rest * c[r_id]

    @staticmethod
    def interpolation(output, radius, factor, b, c):
        return AlgorithmicMethods.interpolation_body(output.data, radius.data, factor, b.data, c.data)

    @staticmethod
    def make_cell_caretaker(idx, cell_start, scheme="default"):
        class CellCaretaker:
            def __init__(self, idx, cell_start, scheme):
                if scheme == "default":
                    if conf.JIT_FLAGS['parallel']:
                        scheme = "counting_sort_parallel"
                    else:
                        scheme = "counting_sort"
                self.scheme = scheme
                if scheme == "counting_sort" or scheme == "counting_sort_parallel":
                    self.tmp_idx = Storage.empty(idx.shape, idx.dtype)
                if scheme == "counting_sort_parallel":
                    self.cell_starts = Storage.empty((numba.config.NUMBA_NUM_THREADS, len(cell_start)), dtype=int)

            def __call__(self, cell_id, cell_idx, cell_start, idx, length):
                if self.scheme == "counting_sort":
                    AlgorithmicMethods._counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data, idx.data, cell_id.data, cell_idx.data, length, cell_start.data)
                elif self.scheme == "counting_sort_parallel":
                    AlgorithmicMethods._parallel_counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data, idx.data, cell_id.data, cell_idx.data, length, cell_start.data,
                        self.cell_starts.data)
                idx.data, self.tmp_idx.data = self.tmp_idx.data, idx.data

        return CellCaretaker(idx, cell_start, scheme)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def moments_body(
            moment_0, moments, n, ex_attr, in_attr, cell_id, idx, length,
            specs_ex_idx, specs_ex_rank, specs_in_idx, specs_in_rank, min_x, max_x, x_attr):
        moment_0[:] = 0
        moments[:, :] = 0
        for i in idx[:length]:
            if min_x < x_attr[i] < max_x:
                moment_0[cell_id[i]] += n[i]
                for k in range(specs_ex_idx.shape[0]):  # TODO #315 (AtomicAdd)
                    moments[k, cell_id[i]] += n[i] * ex_attr[specs_ex_idx[k], i] ** specs_ex_rank[k]
                for k in range(specs_in_idx.shape[0]):
                    moments[specs_ex_idx.shape[0] + k, cell_id[i]] += \
                        n[i] * in_attr[specs_in_idx[k], i] ** specs_in_rank[k]
        for c_id in range(moment_0.shape[0]):
            for k in range(specs_ex_idx.shape[0]):
                moments[k, c_id] = moments[k, c_id] / moment_0[c_id] if moment_0[c_id] != 0 else 0
            for k in range(specs_in_idx.shape[0]):
                moments[specs_ex_idx.shape[0] + k, c_id] = \
                    moments[specs_ex_idx.shape[0] + k, c_id] / moment_0[c_id] if moment_0[c_id] != 0 else 0

    @staticmethod
    def moments(
            moment_0, moments, n, ex_attr, in_attr, cell_id, idx, length,
            specs_ex_idx, specs_ex_rank, specs_in_idx, specs_in_rank, min_x, max_x, x_attr):
        return AlgorithmicMethods.moments_body(
            moment_0.data, moments.data, n.data, ex_attr.data, in_attr.data, cell_id.data,
            idx.data, length, specs_ex_idx.data, specs_ex_rank.data, specs_in_idx.data, specs_in_rank.data,
            min_x, max_x, x_attr.data
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def normalize_body(prob, cell_id, cell_idx, cell_start, norm_factor, dt, dv):
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            sd_num = cell_start[i + 1] - cell_start[i]
            if sd_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = dt / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
        for d in prange(prob.shape[0]):
            prob[d] *= norm_factor[cell_idx[cell_id[d]]]

    @staticmethod
    def normalize(prob, cell_id, cell_idx, cell_start, norm_factor, dt, dv):
        return AlgorithmicMethods.normalize_body(
            prob.data, cell_id.data, cell_idx.data, cell_start.data, norm_factor.data, dt, dv)

    @staticmethod
    @numba.njit(i8(i8[:], i8[:], i8), **{**conf.JIT_FLAGS, **{'parallel': False}})
    def remove_zero_n_or_flagged(multiplicity, idx, length) -> int:
        flag = len(idx)
        new_length = length
        i = 0
        while i < new_length:
            if idx[i] == flag or multiplicity[idx[i]] == 0:
                new_length -= 1
                idx[i] = idx[new_length]
                idx[new_length] = flag
            else:
                i += 1
        return new_length

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'cache': False}})
    def _condensation(
            solver, n_threads, n_cell, cell_start_arg,
            v, particle_temperatures, r_cr, n, vdry, idx, rhod, thd, qv, dv_mean, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order, ripening_flags, RH_max
    ):
        for thread_id in numba.prange(n_threads):
            for i in range(thread_id, n_cell, n_threads):  # TODO #341 at least show that it is not slower :)
                cell_id = cell_order[i]

                cell_start = cell_start_arg[cell_id]
                cell_end = cell_start_arg[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                dthd_dt = (pthd[cell_id] - thd[cell_id]) / dt
                dqv_dt = (pqv[cell_id] - qv[cell_id]) / dt
                rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2
                md = rhod_mean * dv_mean

                qv_new, thd_new, substeps_hint, ripening_flag, RH_max_in_cell = solver(
                    v, particle_temperatures, r_cr, n, vdry,
                    idx[cell_start:cell_end],
                    kappa, thd[cell_id], qv[cell_id], dthd_dt, dqv_dt, md, rhod_mean,
                    rtol_x, rtol_thd, dt, substeps[cell_id]
                )

                substeps[cell_id] = substeps_hint
                ripening_flags[cell_id] += ripening_flag
                RH_max[cell_id] = max(RH_max[cell_id], RH_max_in_cell)

                pqv[cell_id] = qv_new
                pthd[cell_id] = thd_new

    @staticmethod
    @numba.njit(void(i8[:], i8[:], i8[:], i8[:], i8, i8[:]), **conf.JIT_FLAGS)
    def _counting_sort_by_cell_id_and_update_cell_start(new_idx, idx, cell_id, cell_idx, length, cell_start):
        cell_end = cell_start
        # Warning: Assuming len(cell_end) == n_cell+1
        cell_end[:] = 0
        for i in range(length):
            cell_end[cell_idx[cell_id[idx[i]]]] += 1
        for i in range(1, len(cell_end)):
            cell_end[i] += cell_end[i - 1]
        for i in range(length - 1, -1, -1):
            cell_end[cell_idx[cell_id[idx[i]]]] -= 1
            new_idx[cell_end[cell_idx[cell_id[idx[i]]]]] = idx[i]

    @staticmethod
    @numba.njit(void(i8[:], i8[:], i8[:], i8[:], i8, i8[:], i8[:, :]), **conf.JIT_FLAGS)
    def _parallel_counting_sort_by_cell_id_and_update_cell_start(
            new_idx, idx, cell_id, cell_idx, length, cell_start, cell_start_p):
        cell_end_thread = cell_start_p
        # Warning: Assuming len(cell_end) == n_cell+1
        thread_num = cell_end_thread.shape[0]
        for t in prange(thread_num):
            cell_end_thread[t, :] = 0
            for i in range(t * length // thread_num,
                           (t + 1) * length // thread_num if t < thread_num - 1 else length):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] += 1

        cell_start[:] = np.sum(cell_end_thread, axis=0)
        for i in range(1, len(cell_start)):
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
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] -= 1
                new_idx[cell_end_thread[t, cell_idx[cell_id[idx[i]]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]
