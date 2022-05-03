"""
CPU implementation of backend methods for particle collisions
"""
import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_numba.atomic_operations import atomic_add
from PySDM.backends.impl_numba.storage import Storage
from PySDM.backends.impl_numba.warnings import warn
from PySDM.physics.constants import sqrt_pi, sqrt_two


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def pair_indices(i, idx, is_first_in_pair):
    """given permutation array `idx` and `is_first_in_pair` flag array,
    returns indices `j` and `k` of droplets within pair `i`
    such that `j` points to the droplet with higher (or equal) multiplicity
    """
    offset = 1 - is_first_in_pair[2 * i]
    j = idx[2 * i + offset]
    k = idx[2 * i + 1 + offset]
    return j, k


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def flag_zero_multiplicity(j, k, multiplicity, healthy):
    if multiplicity[k] == 0 or multiplicity[j] == 0:
        healthy[0] = 0


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def coalesce(  # pylint: disable=too-many-arguments
    i, j, k, cid, multiplicity, gamma, attributes, coalescence_rate
):
    atomic_add(coalescence_rate, cid, gamma[i] * multiplicity[k])
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


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def break_up(  # pylint: disable=too-many-arguments
    i,
    j,
    k,
    cid,
    multiplicity,
    gamma,
    attributes,
    n_fragment,
    max_multiplicity,
    breakup_rate,
    breakup_rate_deficit,
):  # pylint: disable=too-many-branches
    gamma_tmp = 0
    gamma_deficit = gamma[i]
    overflow_flag = False
    while gamma_deficit > 0:
        if multiplicity[k] > multiplicity[j]:
            j, k = k, j
        tmp1 = 0
        for m in range(int(gamma_deficit)):
            if tmp1 + n_fragment[i] ** m > max_multiplicity:
                atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])
                overflow_flag = True
                break
            tmp1 += n_fragment[i] ** m
            new_n = multiplicity[j] - tmp1 * multiplicity[k]
            gamma_tmp = m + 1
            if new_n < 0:
                gamma_tmp = m
                tmp1 -= n_fragment[i] ** m
                break
        gamma_deficit -= gamma_tmp
        if n_fragment[i] ** gamma_tmp > max_multiplicity:
            break
        tmp2 = n_fragment[i] ** gamma_tmp
        new_n = round(multiplicity[j] - tmp1 * multiplicity[k])

        if tmp2 * multiplicity[k] > max_multiplicity:
            nj = multiplicity[j]
            nk = multiplicity[k]
            atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])
            overflow_flag = True
        elif new_n > 0:
            nj = new_n
            nk = multiplicity[k] * tmp2
            for a in range(0, len(attributes)):
                attributes[a, k] += tmp1 * attributes[a, j]
                attributes[a, k] /= tmp2
        else:  # new_n = 0
            nj = tmp2 * multiplicity[k] / 2
            nk = nj
            for a in range(0, len(attributes)):
                attributes[a, k] += tmp1 * attributes[a, j]
                attributes[a, k] /= tmp2
                attributes[a, j] = attributes[a, k]

        if overflow_flag:
            warn("overflow", __file__)
            break

        atomic_add(breakup_rate, cid, gamma_tmp * multiplicity[k])
        multiplicity[j] = round(nj)
        multiplicity[k] = round(nk)
        factor_j = nj / multiplicity[j]
        factor_k = nk / multiplicity[k]
        for a in range(0, len(attributes)):
            attributes[a, k] *= factor_k
            attributes[a, j] *= factor_j


class CollisionsMethods(BackendMethods):
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    def __adaptive_sdm_end_body(dt_left, n_cell, cell_start):
        end = 0
        for i in range(n_cell - 1, -1, -1):
            if dt_left[i] == 0:
                continue
            end = cell_start[i + 1]
            break
        return end

    def adaptive_sdm_end(self, dt_left, cell_start):
        return self.__adaptive_sdm_end_body(dt_left.data, len(dt_left), cell_start.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments,too-many-locals
    def __adaptive_sdm_gamma_body(
        gamma,
        idx,
        length,
        multiplicity,
        cell_id,
        dt_left,
        dt,
        dt_range,
        is_first_in_pair,
        stats_n_substep,
        stats_dt_min,
    ):
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

    def adaptive_sdm_gamma(
        self,
        *,
        gamma,
        n,
        cell_id,
        dt_left,
        dt,
        dt_range,
        is_first_in_pair,
        stats_n_substep,
        stats_dt_min,
    ):
        return self.__adaptive_sdm_gamma_body(
            gamma.data,
            n.idx.data,
            len(n),
            n.data,
            cell_id.data,
            dt_left.data,
            dt,
            dt_range,
            is_first_in_pair.indicator.data,
            stats_n_substep.data,
            stats_dt_min.data,
        )

    @staticmethod
    # @numba.njit(**conf.JIT_FLAGS)  # note: as of Numba 0.51, np.dot() does not support ints
    def __cell_id_body(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin)

    def cell_id(self, cell_id, cell_origin, strides):
        return self.__cell_id_body(cell_id.data, cell_origin.data, strides.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def __collision_coalescence_body(
        *,
        multiplicity,
        idx,
        length,
        attributes,
        gamma,
        healthy,
        cell_id,
        coalescence_rate,
        is_first_in_pair,
    ):
        for i in numba.prange(  # pylint: disable=not-an-iterable,too-many-nested-blocks
            length // 2
        ):
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)
            coalesce(
                i, j, k, cell_id[i], multiplicity, gamma, attributes, coalescence_rate
            )
            flag_zero_multiplicity(j, k, multiplicity, healthy)

    def collision_coalescence(
        self,
        *,
        multiplicity,
        idx,
        attributes,
        gamma,
        healthy,
        cell_id,
        coalescence_rate,
        is_first_in_pair,
    ):
        self.__collision_coalescence_body(
            multiplicity=multiplicity.data,
            idx=idx.data,
            length=len(idx),
            attributes=attributes.data,
            gamma=gamma.data,
            healthy=healthy.data,
            cell_id=cell_id.data,
            coalescence_rate=coalescence_rate.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def __collision_coalescence_breakup_body(
        *,
        multiplicity,
        idx,
        length,
        attributes,
        gamma,
        rand,
        Ec,
        Eb,
        n_fragment,
        healthy,
        cell_id,
        coalescence_rate,
        breakup_rate,
        breakup_rate_deficit,
        is_first_in_pair,
        max_multiplicity,
    ):
        # pylint: disable=not-an-iterable,too-many-nested-blocks
        for i in numba.prange(length // 2):
            if gamma[i] == 0:
                continue
            bouncing = rand[i] - Ec[i] - Eb[i] > 0
            if bouncing:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)

            if rand[i] - Ec[i] < 0:
                coalesce(
                    i,
                    j,
                    k,
                    cell_id[i],
                    multiplicity,
                    gamma,
                    attributes,
                    coalescence_rate,
                )
            else:
                break_up(
                    i,
                    j,
                    k,
                    cell_id[i],
                    multiplicity,
                    gamma,
                    attributes,
                    n_fragment,
                    max_multiplicity,
                    breakup_rate,
                    breakup_rate_deficit,
                )
            flag_zero_multiplicity(j, k, multiplicity, healthy)

    def collision_coalescence_breakup(
        self,
        *,
        multiplicity,
        idx,
        attributes,
        gamma,
        rand,
        Ec,
        Eb,
        n_fragment,
        healthy,
        cell_id,
        coalescence_rate,
        breakup_rate,
        breakup_rate_deficit,
        is_first_in_pair,
    ):
        max_multiplicity = np.iinfo(multiplicity.data.dtype).max // 2e5
        self.__collision_coalescence_breakup_body(
            multiplicity=multiplicity.data,
            idx=idx.data,
            length=len(idx),
            attributes=attributes.data,
            gamma=gamma.data,
            rand=rand.data,
            Ec=Ec.data,
            Eb=Eb.data,
            n_fragment=n_fragment.data,
            healthy=healthy.data,
            cell_id=cell_id.data,
            coalescence_rate=coalescence_rate.data,
            breakup_rate=breakup_rate.data,
            breakup_rate_deficit=breakup_rate_deficit.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
            max_multiplicity=max_multiplicity,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def __slams_fragmentation_body(n_fragment, probs, rand):
        for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
            probs[i] = 0.0
            n_fragment[i] = 1
            for n in range(22):
                probs[i] += 0.91 * (n + 2) ** (-1.56)
                if rand[i] < probs[i]:
                    n_fragment[i] = n + 2
                    break

    def slams_fragmentation(self, n_fragment, probs, rand):
        self.__slams_fragmentation_body(n_fragment.data, probs.data, rand.data)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def __exp_fragmentation_body(
        *, n_fragment, scale, frag_size, v_max, x_plus_y, rand, vmin, nfmax
    ):
        """
        Exponential PDF
        """
        # TODO #796: add vmin for exp frag
        for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
            frag_size[i] = -scale * np.log(1 - rand[i])
            if frag_size[i] > v_max[i]:
                n_fragment[i] = 1
            elif frag_size[i] < vmin:
                n_fragment[i] = 1
            else:
                n_fragment[i] = x_plus_y[i] / frag_size[i]
            if nfmax is not None:
                n_fragment[i] = min(n_fragment[i], nfmax)

    def exp_fragmentation(
        self, *, n_fragment, scale, frag_size, v_max, x_plus_y, rand, vmin, nfmax
    ):
        self.__exp_fragmentation_body(
            n_fragment=n_fragment.data,
            scale=scale,
            frag_size=frag_size.data,
            v_max=v_max.data,
            x_plus_y=x_plus_y.data,
            rand=rand.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def __feingold1988_fragmentation_body(
        *, n_fragment, scale, frag_size, v_max, x_plus_y, rand, fragtol, vmin, nfmax
    ):
        """
        Scaled exponential PDF
        """
        for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
            log_arg = max(1 - rand[i] * scale / x_plus_y[i], fragtol)
            frag_size[i] = -scale * np.log(log_arg)
            if frag_size[i] > v_max[i]:
                n_fragment[i] = 1
            elif frag_size[i] < vmin:
                n_fragment[i] = 1
            else:
                n_fragment[i] = x_plus_y[i] / frag_size[i]
            if nfmax is not None:
                n_fragment[i] = min(n_fragment[i], nfmax)

    def feingold1988_fragmentation(
        self,
        *,
        n_fragment,
        scale,
        frag_size,
        v_max,
        x_plus_y,
        rand,
        fragtol,
        vmin,
        nfmax,
    ):
        self.__feingold1988_fragmentation_body(
            n_fragment=n_fragment.data,
            scale=scale,
            frag_size=frag_size.data,
            v_max=v_max.data,
            x_plus_y=x_plus_y.data,
            rand=rand.data,
            fragtol=fragtol,
            vmin=vmin,
            nfmax=nfmax,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def __gauss_fragmentation_body(*, n_fragment, mu, scale, frag_size, r_max, rand):
        """
        Gaussian PDF
        CDF = erf(x); approximate as erf(x) ~ tanh(ax) with a = 2/sqrt(pi) as in Vedder 1987
        """
        for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
            frag_size[i] = mu + sqrt_pi * sqrt_two * scale / 4 * np.log(
                (1 + rand[i]) / (1 - rand[i])
            )
            if frag_size[i] > r_max[i]:
                n_fragment[i] = 1.0
            else:
                n_fragment[i] = r_max[i] / frag_size[i]

    def gauss_fragmentation(self, *, n_fragment, mu, scale, frag_size, r_max, rand):
        self.__gauss_fragmentation_body(
            n_fragment=n_fragment.data,
            mu=mu,
            scale=scale,
            frag_size=frag_size.data,
            r_max=r_max.data,
            rand=rand.data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def __ll1982_fragmentation_body(
        n_fragment, probs, rand
    ):  # pylint: disable=unused-argument
        for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
            probs[i] = 0.0
            n_fragment[i] = 1

            # first consider filament breakup

    def ll1982_fragmentation(self, n_fragment, probs, rand):
        self.__ll1982_fragmentation_body(n_fragment.data, probs.data, rand.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments
    def __compute_gamma_body(
        gamma,
        rand,
        idx,
        length,
        multiplicity,
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
    ):
        """
        return in "gamma" array gamma (see: http://doi.org/10.1002/qj.441, section 5)
        formula:
        gamma = floor(prob) + 1 if rand <  prob - floor(prob)
              = floor(prob)     if rand >= prob - floor(prob)
        """
        for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
            gamma[i] = np.ceil(gamma[i] - rand[i])

            no_collision = gamma[i] == 0
            if no_collision:
                continue

            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = multiplicity[j] // multiplicity[k]
            g = min(int(gamma[i]), prop)
            cid = cell_id[j]
            atomic_add(collision_rate, cid, g * multiplicity[k])
            atomic_add(
                collision_rate_deficit, cid, (int(gamma[i]) - g) * multiplicity[k]
            )
            gamma[i] = g

    def compute_gamma(
        self,
        *,
        gamma,
        rand,
        multiplicity,
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
    ):
        return self.__compute_gamma_body(
            gamma.data,
            rand.data,
            multiplicity.idx.data,
            len(multiplicity),
            multiplicity.data,
            cell_id.data,
            collision_rate_deficit.data,
            collision_rate.data,
            is_first_in_pair.indicator.data,
        )

    @staticmethod
    def make_cell_caretaker(idx, cell_start, scheme="default"):
        class CellCaretaker:  # pylint: disable=too-few-public-methods
            def __init__(self, idx, cell_start, scheme):
                if scheme == "default":
                    if conf.JIT_FLAGS["parallel"]:
                        scheme = "counting_sort_parallel"
                    else:
                        scheme = "counting_sort"
                self.scheme = scheme
                if scheme in ("counting_sort", "counting_sort_parallel"):
                    self.tmp_idx = Storage.empty(idx.shape, idx.dtype)
                if scheme == "counting_sort_parallel":
                    self.cell_starts = Storage.empty(
                        (
                            numba.config.NUMBA_NUM_THREADS,  # pylint: disable=no-member
                            len(cell_start),
                        ),
                        dtype=int,
                    )

            def __call__(self, cell_id, cell_idx, cell_start, idx):
                length = len(idx)
                if self.scheme == "counting_sort":
                    CollisionsMethods._counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data,
                        idx.data,
                        cell_id.data,
                        cell_idx.data,
                        length,
                        cell_start.data,
                    )
                elif self.scheme == "counting_sort_parallel":
                    CollisionsMethods._parallel_counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data,
                        idx.data,
                        cell_id.data,
                        cell_idx.data,
                        length,
                        cell_start.data,
                        self.cell_starts.data,
                    )
                idx.data, self.tmp_idx.data = self.tmp_idx.data, idx.data

        return CellCaretaker(idx, cell_start, scheme)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    # pylint: disable=too-many-arguments
    def __normalize_body(
        prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv
    ):
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            sd_num = cell_start[i + 1] - cell_start[i]
            if sd_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = (
                    timestep / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
                )
        for d in numba.prange(prob.shape[0]):  # pylint: disable=not-an-iterable
            prob[d] *= norm_factor[cell_idx[cell_id[d]]]

    # pylint: disable=too-many-arguments
    def normalize(self, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
        return self.__normalize_body(
            prob.data,
            cell_id.data,
            cell_idx.data,
            cell_start.data,
            norm_factor.data,
            timestep,
            dv,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
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
        new_idx, idx, cell_id, cell_idx, length, cell_start, cell_start_p
    ):
        cell_end_thread = cell_start_p
        # Warning: Assuming len(cell_end) == n_cell+1
        thread_num = cell_end_thread.shape[0]
        for t in numba.prange(thread_num):  # pylint: disable=not-an-iterable
            cell_end_thread[t, :] = 0
            for i in range(
                t * length // thread_num,
                (t + 1) * length // thread_num if t < thread_num - 1 else length,
            ):
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
                (t + 1) * length // thread_num - 1
                if t < thread_num - 1
                else length - 1,
                t * length // thread_num - 1,
                -1,
            ):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] -= 1
                new_idx[cell_end_thread[t, cell_idx[cell_id[idx[i]]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]
