"""
CPU implementation of backend methods for particle collisions
"""
import numba
import numpy as np
from PySDM.physics.constants import sqrt_pi, sqrt_two
from PySDM.backends.numba import conf
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_numba.storage import Storage
from PySDM.backends.impl_common.backend_methods import BackendMethods


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def pair_indices(i, idx, is_first_in_pair):
    offset = 1 - is_first_in_pair[2 * i]
    j = idx[2 * i + offset]
    k = idx[2 * i + 1 + offset]
    return j, k


class AlgorithmicMethods(BackendMethods):
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def adaptive_sdm_end_body(dt_left, n_cell, cell_start):
        end = 0
        for i in range(n_cell - 1, -1, -1):
            if dt_left[i] == 0:
                continue
            end = cell_start[i + 1]
            break
        return end

    def adaptive_sdm_end(self, dt_left, cell_start):
        return self.adaptive_sdm_end_body(
            dt_left.data, len(dt_left), cell_start.data
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments,too-many-locals
    def adaptive_sdm_gamma_body(gamma, idx, length, multiplicity, cell_id, dt_left, dt,
                                dt_range, is_first_in_pair,
                                stats_n_substep, stats_dt_min):
        dt_todo = np.empty_like(dt_left)
        for cid in numba.prange(len(dt_todo)):  # pylint: disable=not-an-iterable
            dt_todo[cid] = min(dt_left[cid], dt_range[1])
        for i in range(length // 2):  # TODO #571
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = multiplicity[j] // multiplicity[k]
            dt_optimal = dt * prop / gamma[i]
            cid = cell_id[j]
            dt_optimal = max(dt_optimal, dt_range[0])
            dt_todo[cid] = min(dt_todo[cid], dt_optimal)
            stats_dt_min[cid] = min(stats_dt_min[cid], dt_optimal)
        for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
            if gamma[i] == 0:
                continue
            j, _ = pair_indices(i, idx, is_first_in_pair)
            gamma[i] *= dt_todo[cell_id[j]] / dt
        for cid in numba.prange(len(dt_todo)):  # pylint: disable=not-an-iterable
            dt_left[cid] -= dt_todo[cid]
            if dt_todo[cid] > 0:
                stats_n_substep[cid] += 1

    # pylint: disable=too-many-arguments
    def adaptive_sdm_gamma(self, gamma, n, cell_id, dt_left, dt, dt_range, is_first_in_pair,
                           stats_n_substep, stats_dt_min):
        return self.adaptive_sdm_gamma_body(
            gamma.data, n.idx.data, len(n), n.data, cell_id.data,
            dt_left.data, dt, dt_range, is_first_in_pair.indicator.data,
            stats_n_substep.data, stats_dt_min.data)

    @staticmethod
    # @numba.njit(**conf.JIT_FLAGS)  # note: as of Numba 0.51, np.dot() does not support ints
    def cell_id_body(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin)

    def cell_id(self, cell_id, cell_origin, strides):
        return self.cell_id_body(cell_id.data, cell_origin.data, strides.data)
        
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def collision_body(multiplicity, idx, length, attributes, gamma, rand, dyn, Ec, Eb, n_fragment, healthy, cell_id,
                        coalescence_rate, breakup_rate, is_first_in_pair):
        for i in numba.prange(length // 2):
            cid = cell_id[i]
            dyn[i] = rand[i] - Ec[i] - Eb[i]

            if dyn[i] > 0: # bouncing
                continue

            dyn[i] = rand[i] - Ec[i]
            j, k = pair_indices(i, idx, is_first_in_pair)
                
            if dyn[i] < 0: # coalescence
                new_n = multiplicity[j] - gamma[i] * multiplicity[k]
                if new_n > 0:
                    multiplicity[j] = new_n
                    for a in range(0, len(attributes)):
                        attributes[a, k] += gamma[i] * attributes[a, j]
                else:  # new_n == 0
                    multiplicity[j] = multiplicity[k] // 2
                    multiplicity[k] = multiplicity[k] - multiplicity[j]
                    for a in range(0, len(attributes)):
                        attributes[a, j] = gamma[i] * attributes[a, j] + attributes[a, k]
                        attributes[a, k] = attributes[a, j]
                if multiplicity[k] == 0 or multiplicity[j] == 0:
                    healthy[0] = 0

                coalescence_rate[cid] += gamma[i] * multiplicity[k]
                    
            else: # breakup
                new_n = multiplicity[j] - gamma[i] * multiplicity[k]
                # perform rounding to keep multiplicity[k] as integer
                n_fragment[i] = int(multiplicity[k] * n_fragment[i]) / multiplicity[k]
                if new_n > 0:
                    multiplicity[j] = new_n
                    multiplicity[k] = int(multiplicity[k] * n_fragment[i])
                    for a in range(0, len(attributes)):
                        attributes[a, k] += gamma[i] * attributes[a, j]
                        attributes[a, k] /= n_fragment[i]

                else:  # new_n == 0
                    multiplicity[j] = (n_fragment[i] * multiplicity[k]) // 2
                    multiplicity[k] = n_fragment[i] * multiplicity[k] - multiplicity[j]
                    for a in range(0, len(attributes)):
                        attributes[a, j] = (gamma[i] * attributes[a, j] + attributes[a, k])/n_fragment[i]
                        attributes[a, k] = attributes[a, j]
                if multiplicity[k] == 0 or multiplicity[j] == 0:
                    healthy[0] = 0

                breakup_rate[cid] += gamma[i] * multiplicity[k]

    @staticmethod
    def collision(n, idx, length, attributes, gamma, rand, dyn, Ec, Eb, n_fragment, healthy, cell_id,
                        coalescence_rate, breakup_rate, is_first_in_pair):
        AlgorithmicMethods.collision_body(n.data, idx.data, length,
                                            attributes.data, gamma.data, rand.data, dyn.data, Ec.data, Eb.data, 
                                            n_fragment.data, healthy.data, cell_id.data, coalescence_rate.data, 
                                            breakup_rate.data, is_first_in_pair.indicator.data)
        
    @numba.njit(**{**conf.JIT_FLAGS})
    def slams_fragmentation_body(n_fragment, probs, rand):
        for i in numba.prange(len(n_fragment)):
            probs[i] = 0.0
            n_fragment[i] = 1
            for n in range(22):
                probs[i] += 0.91 * (n + 2)**(-1.56)
                if (rand[i] < probs[i]):
                    n_fragment[i] = n + 2
                    break
    
    @staticmethod
    def slams_fragmentation(n_fragment, probs, rand):
        AlgorithmicMethods.slams_fragmentation_body(n_fragment.data, probs.data, rand.data)

    '''
    Exponential PDF
    '''
    @numba.njit(**{**conf.JIT_FLAGS})
    def exp_fragmentation_body(n_fragment, scale, frag_size, r_max, rand):
        for i in numba.prange(len(n_fragment)):
            frag_size[i] = -scale * np.log(1-rand[i])
            if (frag_size[i] > r_max[i]):
                n_fragment[i] = 1
            else:
                n_fragment[i] = r_max[i] / frag_size[i]
    
    @staticmethod
    def exp_fragmentation(n_fragment, scale, frag_size, r_max, rand):
        AlgorithmicMethods.exp_fragmentation_body(n_fragment.data, scale, frag_size.data, r_max.data, rand.data)

    '''
    Gaussian PDF
    CDF = erf(x); approximate as erf(x) ~ tanh(ax) with a = 2/sqrt(pi) as in Vedder 1987
    '''
    @numba.njit(**{**conf.JIT_FLAGS})
    def gauss_fragmentation_body(n_fragment, mu, scale, frag_size, r_max, rand):
        for i in numba.prange(len(n_fragment)):
            frag_size[i] = mu + sqrt_pi * sqrt_two * scale / 4 * np.log((1 + rand[i])/(1 - rand[i]))
            if (frag_size[i] > r_max[i]):
                n_fragment[i] = 1.0
            else:
                n_fragment[i] = r_max[i] / frag_size[i]
    
    @staticmethod
    def gauss_fragmentation(n_fragment, mu, scale, frag_size, r_max, rand):
        AlgorithmicMethods.gauss_fragmentation_body(n_fragment.data, mu, scale, frag_size.data, r_max.data, rand.data)

    # Emily: Low and List 1982 fragmentation function
    @numba.njit(**{**conf.JIT_FLAGS})
    def ll1982_fragmentation_body(n_fragment, probs, rand):
        for i in numba.prange(len(n_fragment)):
            probs[i] = 0.0
            n_fragment[i] = 1
            
            # first consider filament breakup
    
    @staticmethod
    def ll1982_fragmentation(n_fragment, probs, rand):
        AlgorithmicMethods.ll1982_fragmentation_body(n_fragment.data, probs.data, rand.data)
        
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments
    def compute_gamma_body(gamma, rand, idx, length, multiplicity, cell_id,
                           collision_rate_deficit, collision_rate, is_first_in_pair):

        """
        return in "gamma" array gamma (see: http://doi.org/10.1002/qj.441, section 5)
        formula:
        gamma = floor(prob) + 1 if rand <  prob - floor(prob)
              = floor(prob)     if rand >= prob - floor(prob)
        """
        for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
            gamma[i] = np.ceil(gamma[i] - rand[i])

            # (4) No collision
            if gamma[i] == 0:
                continue

            # (5) Successful collision
            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = multiplicity[j] // multiplicity[k]
            g = min(int(gamma[i]), prop)
            cid = cell_id[j]
            # compute the number of collisions
            collision_rate[cid] += g * multiplicity[k]
            collision_rate_deficit[cid] += (int(gamma[i]) - g) * multiplicity[k]
            gamma[i] = g

    # pylint: disable=too-many-arguments
    def compute_gamma(self, gamma, rand, multiplicity, cell_id,
                      collision_rate_deficit, collision_rate, is_first_in_pair):
        return self.compute_gamma_body(
            gamma.data, rand.data, multiplicity.idx.data, len(multiplicity), multiplicity.data,
            cell_id.data,
            collision_rate_deficit.data, collision_rate.data, is_first_in_pair.indicator.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments,too-many-locals
    def linear_collection_efficiency_body(
            params, output, radii, is_first_in_pair, idx, length, unit
    ):
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        output[:] = 0
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                if radii[idx[i]] > radii[idx[i + 1]]:
                    r = radii[idx[i]] / unit
                    r_s = radii[idx[i + 1]] / unit
                else:
                    r = radii[idx[i + 1]] / unit
                    r_s = radii[idx[i]] / unit
                p = r_s / r
                if p not in (0, 1):
                    G = (G1 / r) ** Mg + G2 + G3 * r
                    Gp = (1 - p) ** G
                    if Gp != 0:
                        D = D1 / r ** D2
                        E = E1 / r ** E2
                        F = (F1 / r) ** Mf + F2
                        output[i // 2] = A + B * p + D / p ** F + E / Gp
                        output[i // 2] = max(0, output[i // 2])

    def linear_collection_efficiency(self, params, output, radii, is_first_in_pair, unit):
        return self.linear_collection_efficiency_body(
            params, output.data, radii.data, is_first_in_pair.indicator.data,
            radii.idx.data, len(is_first_in_pair), unit)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def interpolation_body(output, radius, factor, b, c):
        for i in numba.prange(len(radius)):  # pylint: disable=not-an-iterable
            r_id = int(factor * radius[i])
            r_rest = ((factor * radius[i]) % 1) / factor
            output[i] = b[r_id] + r_rest * c[r_id]

    def interpolation(self, output, radius, factor, b, c):
        return self.interpolation_body(
            output.data, radius.data, factor, b.data, c.data
        )

    @staticmethod
    def make_cell_caretaker(idx, cell_start, scheme="default"):
        class CellCaretaker:  # pylint: disable=too-few-public-methods
            def __init__(self, idx, cell_start, scheme):
                if scheme == "default":
                    if conf.JIT_FLAGS['parallel']:
                        scheme = "counting_sort_parallel"
                    else:
                        scheme = "counting_sort"
                self.scheme = scheme
                if scheme in ("counting_sort", "counting_sort_parallel"):
                    self.tmp_idx = Storage.empty(idx.shape, idx.dtype)
                if scheme == "counting_sort_parallel":
                    self.cell_starts = Storage.empty(
                        (numba.config.NUMBA_NUM_THREADS, len(cell_start)),  # pylint: disable=no-member
                        dtype=int
                    )

            def __call__(self, cell_id, cell_idx, cell_start, idx):
                length = len(idx)
                if self.scheme == "counting_sort":
                    AlgorithmicMethods._counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data, idx.data, cell_id.data,
                        cell_idx.data, length, cell_start.data)
                elif self.scheme == "counting_sort_parallel":
                    AlgorithmicMethods._parallel_counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data, idx.data, cell_id.data, cell_idx.data,
                        length, cell_start.data, self.cell_starts.data)
                idx.data, self.tmp_idx.data = self.tmp_idx.data, idx.data

        return CellCaretaker(idx, cell_start, scheme)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    # pylint: disable=too-many-arguments
    def normalize_body(prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            sd_num = cell_start[i + 1] - cell_start[i]
            if sd_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = timestep / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
        for d in numba.prange(prob.shape[0]):  # pylint: disable=not-an-iterable
            prob[d] *= norm_factor[cell_idx[cell_id[d]]]

    # pylint: disable=too-many-arguments
    def normalize(self, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
        return self.normalize_body(
            prob.data, cell_id.data, cell_idx.data, cell_start.data,
            norm_factor.data, timestep, dv)


    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
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
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments
    def _counting_sort_by_cell_id_and_update_cell_start(
            new_idx, idx, cell_id, cell_idx, length, cell_start
    ):
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
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments
    def _parallel_counting_sort_by_cell_id_and_update_cell_start(
            new_idx, idx, cell_id, cell_idx, length, cell_start, cell_start_p):
        cell_end_thread = cell_start_p
        # Warning: Assuming len(cell_end) == n_cell+1
        thread_num = cell_end_thread.shape[0]
        for t in numba.prange(thread_num):  # pylint: disable=not-an-iterable
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

        for t in numba.prange(thread_num):  # pylint: disable=not-an-iterable
            for i in range(
                (t + 1) * length // thread_num - 1 if t < thread_num - 1 else length - 1,
                t * length // thread_num - 1,
                -1
            ):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] -= 1
                new_idx[cell_end_thread[t, cell_idx[cell_id[idx[i]]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]
