"""
CPU implementation of backend methods for particle collisions
"""

from functools import cached_property
import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_numba.atomic_operations import atomic_add
from PySDM.backends.impl_numba.storage import Storage
from PySDM.backends.impl_numba.warnings import warn


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def pair_indices(i, idx, is_first_in_pair, prob_like):
    """given permutation array `idx` and `is_first_in_pair` flag array,
    returns indices `j` and `k` of droplets within pair `i` and a `skip_pair` flag,
    `j` points to the droplet that is first in pair (higher or equal multiplicity)
    output is valid only if `2*i` or `2*i+1` points to a valid pair start index (within one cell)
    otherwise the `skip_pair` flag is set to True and returned `j` & `k` indices are set to -1.
    In addition, the `prob_like` array is checked for zeros at position `i`, in which case
    the `skip_pair` is also set to `True`
    """
    skip_pair = False

    if prob_like[i] == 0:
        skip_pair = True
        j, k = -1, -1
    else:
        offset = 1 - is_first_in_pair[2 * i]
        j = idx[2 * i + offset]
        k = idx[2 * i + 1 + offset]
    return j, k, skip_pair


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
        for a in range(len(attributes)):
            attributes[a, k] += gamma[i] * attributes[a, j]
    else:  # new_n == 0
        multiplicity[j] = multiplicity[k] // 2
        multiplicity[k] = multiplicity[k] - multiplicity[j]
        for a in range(len(attributes)):
            attributes[a, j] = gamma[i] * attributes[a, j] + attributes[a, k]
            attributes[a, k] = attributes[a, j]


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def compute_transfer_multiplicities(
    gamma, j, k, multiplicity, particle_mass, fragment_mass_i, max_multiplicity
):  # pylint: disable=too-many-arguments
    overflow_flag = False
    gamma_j_k = 0
    take_from_j_test = multiplicity[k]
    take_from_j = 0
    new_mult_k_test = (
        (particle_mass[j] + particle_mass[k]) / fragment_mass_i
    ) * multiplicity[k]
    new_mult_k = multiplicity[k]
    for m in range(int(gamma)):
        # check for overflow of multiplicity
        if new_mult_k_test > max_multiplicity:
            overflow_flag = True
            break

        # check for new_n >= 0
        if take_from_j_test > multiplicity[j]:
            break

        take_from_j = take_from_j_test
        new_mult_k = new_mult_k_test
        gamma_j_k = m + 1

        take_from_j_test += new_mult_k_test
        new_mult_k_test = (
            new_mult_k_test * (particle_mass[j] / fragment_mass_i) + new_mult_k_test
        )

    return take_from_j, new_mult_k, gamma_j_k, overflow_flag


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def get_new_multiplicities_and_update_attributes(
    j, k, attributes, multiplicity, take_from_j, new_mult_k
):  # pylint: disable=too-many-arguments
    for a in range(len(attributes)):
        attributes[a, k] *= multiplicity[k]
        attributes[a, k] += take_from_j * attributes[a, j]
        attributes[a, k] /= new_mult_k

    if multiplicity[j] > take_from_j:
        nj = multiplicity[j] - take_from_j
        nk = new_mult_k

    else:  # take_from_j == multiplicity[j]
        nj = new_mult_k / 2
        nk = nj
        for a in range(len(attributes)):
            attributes[a, j] = attributes[a, k]
    return nj, nk


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def round_multiplicities_to_ints_and_update_attributes(
    j,
    k,
    nj,
    nk,
    attributes,
    multiplicity,
):  # pylint: disable=too-many-arguments
    multiplicity[j] = max(round(nj), 1)
    multiplicity[k] = max(round(nk), 1)
    factor_j = nj / multiplicity[j]
    factor_k = nk / multiplicity[k]
    for a in range(len(attributes)):
        attributes[a, k] *= factor_k
        attributes[a, j] *= factor_j


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def break_up(  # pylint: disable=too-many-arguments,c,too-many-locals
    i,
    j,
    k,
    cid,
    multiplicity,
    gamma,
    attributes,
    fragment_mass,
    max_multiplicity,
    breakup_rate,
    breakup_rate_deficit,
    warn_overflows,
    particle_mass,
):  # breakup0 guarantees take_from_j <= multiplicity[j]
    take_from_j, new_mult_k, gamma_j_k, overflow_flag = compute_transfer_multiplicities(
        gamma[i],
        j,
        k,
        multiplicity,
        particle_mass,
        fragment_mass[i],
        max_multiplicity,
    )
    gamma_deficit = gamma[i] - gamma_j_k

    # breakup1 also handles new_n[j] == 0 case via splitting
    nj, nk = get_new_multiplicities_and_update_attributes(
        j, k, attributes, multiplicity, take_from_j, new_mult_k
    )

    atomic_add(breakup_rate, cid, gamma_j_k * multiplicity[k])
    atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])

    # breakup2 also guarantees that no multiplicities are set to 0
    round_multiplicities_to_ints_and_update_attributes(
        j, k, nj, nk, attributes, multiplicity
    )
    if overflow_flag and warn_overflows:
        warn("overflow", __file__)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def break_up_while(
    i,
    j,
    k,
    cid,
    multiplicity,
    gamma,
    attributes,
    fragment_mass,
    max_multiplicity,
    breakup_rate,
    breakup_rate_deficit,
    warn_overflows,
    particle_mass,
):  # pylint: disable=too-many-arguments,unused-argument,too-many-locals
    gamma_deficit = gamma[i]
    overflow_flag = False
    while gamma_deficit > 0:
        if multiplicity[k] == multiplicity[j]:
            take_from_j = multiplicity[j]
            new_mult_k = (
                (particle_mass[j] + particle_mass[k])
                / fragment_mass[i]
                * multiplicity[k]
            )

            # check for overflow
            if new_mult_k > max_multiplicity:
                atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])
                overflow_flag = True
                break
            gamma_j_k = gamma_deficit

        else:
            if multiplicity[k] > multiplicity[j]:
                j, k = k, j
            (
                take_from_j,
                new_mult_k,
                gamma_j_k,
                overflow_flag,
            ) = compute_transfer_multiplicities(
                gamma_deficit,
                j,
                k,
                multiplicity,
                particle_mass,
                fragment_mass[i],
                max_multiplicity,
            )

        nj, nk = get_new_multiplicities_and_update_attributes(
            j, k, attributes, multiplicity, take_from_j, new_mult_k
        )

        atomic_add(breakup_rate, cid, gamma_j_k * multiplicity[k])
        gamma_deficit -= gamma_j_k
        round_multiplicities_to_ints_and_update_attributes(
            j, k, nj, nk, attributes, multiplicity
        )

    atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])

    if overflow_flag and warn_overflows:
        warn("overflow", __file__)


class CollisionsMethods(BackendMethods):
    @cached_property
    def _collision_coalescence_breakup_body(self):
        _break_up = break_up_while if self.formulae.handle_all_breakups else break_up

        @numba.njit(**self.default_jit_flags)
        def body(
            *,
            multiplicity,
            idx,
            length,
            attributes,
            gamma,
            rand,
            Ec,
            Eb,
            fragment_mass,
            healthy,
            cell_id,
            coalescence_rate,
            breakup_rate,
            breakup_rate_deficit,
            is_first_in_pair,
            max_multiplicity,
            warn_overflows,
            particle_mass,
        ):
            # pylint: disable=not-an-iterable,too-many-nested-blocks,too-many-locals
            for i in numba.prange(length // 2):
                j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, gamma)
                if skip_pair:
                    continue
                bouncing = rand[i] - (Ec[i] + (1 - Ec[i]) * (Eb[i])) > 0
                if bouncing:
                    continue

                if rand[i] - Ec[i] < 0:
                    coalesce(
                        i,
                        j,
                        k,
                        cell_id[j],
                        multiplicity,
                        gamma,
                        attributes,
                        coalescence_rate,
                    )
                else:
                    _break_up(
                        i,
                        j,
                        k,
                        cell_id[j],
                        multiplicity,
                        gamma,
                        attributes,
                        fragment_mass,
                        max_multiplicity,
                        breakup_rate,
                        breakup_rate_deficit,
                        warn_overflows,
                        particle_mass,
                    )
                flag_zero_multiplicity(j, k, multiplicity, healthy)

        return body

    @cached_property
    def _adaptive_sdm_end_body(self):
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def body(dt_left, n_cell, cell_start):
            end = 0
            for i in range(n_cell - 1, -1, -1):
                if dt_left[i] == 0:
                    continue
                end = cell_start[i + 1]
                break
            return end

        return body

    def adaptive_sdm_end(self, dt_left, cell_start):
        return self._adaptive_sdm_end_body(dt_left.data, len(dt_left), cell_start.data)

    @cached_property
    def _scale_prob_for_adaptive_sdm_gamma_body(self):
        @numba.njit(**self.default_jit_flags)
        # pylint: disable=too-many-arguments,too-many-locals
        def body(
            prob,
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
                j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, prob)
                if skip_pair:
                    continue
                prop = multiplicity[j] // multiplicity[k]
                dt_optimal = dt * prop / prob[i]
                cid = cell_id[j]
                dt_optimal = max(dt_optimal, dt_range[0])
                dt_todo[cid] = min(dt_todo[cid], dt_optimal)
                stats_dt_min[cid] = min(stats_dt_min[cid], dt_optimal)
            for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
                j, _, skip_pair = pair_indices(i, idx, is_first_in_pair, prob)
                if skip_pair:
                    continue
                prob[i] *= dt_todo[cell_id[j]] / dt
            for cid in numba.prange(len(dt_todo)):  # pylint: disable=not-an-iterable
                dt_left[cid] -= dt_todo[cid]
                if dt_todo[cid] > 0:
                    stats_n_substep[cid] += 1

        return body

    def scale_prob_for_adaptive_sdm_gamma(
        self,
        *,
        prob,
        multiplicity,
        cell_id,
        dt_left,
        dt,
        dt_range,
        is_first_in_pair,
        stats_n_substep,
        stats_dt_min,
    ):
        return self._scale_prob_for_adaptive_sdm_gamma_body(
            prob.data,
            multiplicity.idx.data,
            len(multiplicity),
            multiplicity.data,
            cell_id.data,
            dt_left.data,
            dt,
            dt_range,
            is_first_in_pair.indicator.data,
            stats_n_substep.data,
            stats_dt_min.data,
        )

    @cached_property
    def _cell_id_body(self):
        # @numba.njit(**conf.JIT_FLAGS)  # note: as of Numba 0.51, np.dot() does not support ints
        def body(cell_id, cell_origin, strides):
            cell_id[:] = np.dot(strides, cell_origin)

        return body

    def cell_id(self, cell_id, cell_origin, strides):
        return self._cell_id_body(cell_id.data, cell_origin.data, strides.data)

    @cached_property
    def _collision_coalescence_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(
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
            for (
                i
            ) in numba.prange(  # pylint: disable=not-an-iterable,too-many-nested-blocks
                length // 2
            ):
                j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, gamma)
                if skip_pair:
                    continue
                coalesce(
                    i,
                    j,
                    k,
                    cell_id[j],
                    multiplicity,
                    gamma,
                    attributes,
                    coalescence_rate,
                )
                flag_zero_multiplicity(j, k, multiplicity, healthy)

        return body

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
        self._collision_coalescence_body(
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
        fragment_mass,
        healthy,
        cell_id,
        coalescence_rate,
        breakup_rate,
        breakup_rate_deficit,
        is_first_in_pair,
        warn_overflows,
        particle_mass,
        max_multiplicity,
    ):
        # pylint: disable=too-many-locals
        self._collision_coalescence_breakup_body(
            multiplicity=multiplicity.data,
            idx=idx.data,
            length=len(idx),
            attributes=attributes.data,
            gamma=gamma.data,
            rand=rand.data,
            Ec=Ec.data,
            Eb=Eb.data,
            fragment_mass=fragment_mass.data,
            healthy=healthy.data,
            cell_id=cell_id.data,
            coalescence_rate=coalescence_rate.data,
            breakup_rate=breakup_rate.data,
            breakup_rate_deficit=breakup_rate_deficit.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
            max_multiplicity=max_multiplicity,
            warn_overflows=warn_overflows,
            particle_mass=particle_mass.data,
        )

    @cached_property
    def _compute_gamma_body(self):
        @numba.njit(**self.default_jit_flags)
        # pylint: disable=too-many-arguments,too-many-locals
        def body(
            prob,
            rand,
            idx,
            length,
            multiplicity,
            cell_id,
            collision_rate_deficit,
            collision_rate,
            is_first_in_pair,
            out,
        ):
            """
            return in "out" array gamma (see: http://doi.org/10.1002/qj.441, section 5)
            formula:
            gamma = floor(prob) + 1 if rand <  prob - floor(prob)
                  = floor(prob)     if rand >= prob - floor(prob)

            out may point to the same array as prob
            """
            for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
                out[i] = np.ceil(prob[i] - rand[i])
                j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, out)
                if skip_pair:
                    continue
                prop = multiplicity[j] // multiplicity[k]
                g = min(int(out[i]), prop)
                cid = cell_id[j]
                atomic_add(collision_rate, cid, g * multiplicity[k])
                atomic_add(
                    collision_rate_deficit, cid, (int(out[i]) - g) * multiplicity[k]
                )
                out[i] = g

        return body

    def compute_gamma(
        self,
        *,
        prob,
        rand,
        multiplicity,
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
        out,
    ):
        return self._compute_gamma_body(
            prob.data,
            rand.data,
            multiplicity.idx.data,
            len(multiplicity),
            multiplicity.data,
            cell_id.data,
            collision_rate_deficit.data,
            collision_rate.data,
            is_first_in_pair.indicator.data,
            out.data,
        )

    @staticmethod
    def make_cell_caretaker(idx_shape, idx_dtype, cell_start_len, scheme="default"):
        class CellCaretaker:  # pylint: disable=too-few-public-methods
            def __init__(self, idx_shape, idx_dtype, cell_start_len, scheme):
                if scheme == "default":
                    if conf.JIT_FLAGS["parallel"]:
                        scheme = "counting_sort_parallel"
                    else:
                        scheme = "counting_sort"
                self.scheme = scheme
                if scheme in ("counting_sort", "counting_sort_parallel"):
                    self.tmp_idx = Storage.empty(idx_shape, idx_dtype)
                if scheme == "counting_sort_parallel":
                    self.cell_starts = Storage.empty(
                        (
                            numba.config.NUMBA_NUM_THREADS,  # pylint: disable=no-member
                            cell_start_len,
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

        return CellCaretaker(idx_shape, idx_dtype, cell_start_len, scheme)

    @cached_property
    def _normalize_body(self):
        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        # pylint: disable=too-many-arguments
        def body(prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
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

        return body

    # pylint: disable=too-many-arguments
    def normalize(self, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
        return self._normalize_body(
            prob.data,
            cell_id.data,
            cell_idx.data,
            cell_start.data,
            norm_factor.data,
            timestep,
            dv,
        )

    @cached_property
    def remove_zero_n_or_flagged(self):
        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(multiplicity, idx, length) -> int:
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

        return body

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
                (
                    (t + 1) * length // thread_num - 1
                    if t < thread_num - 1
                    else length - 1
                ),
                t * length // thread_num - 1,
                -1,
            ):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] -= 1
                new_idx[cell_end_thread[t, cell_idx[cell_id[idx[i]]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]

    @cached_property
    def _linear_collection_efficiency_body(self):
        @numba.njit(**self.default_jit_flags)
        # pylint: disable=too-many-arguments,too-many-locals
        def body(params, output, radii, is_first_in_pair, idx, length, unit):
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
                            D = D1 / r**D2
                            E = E1 / r**E2
                            F = (F1 / r) ** Mf + F2
                            output[i // 2] = A + B * p + D / p**F + E / Gp
                            output[i // 2] = max(0, output[i // 2])

        return body

    def linear_collection_efficiency(
        self, *, params, output, radii, is_first_in_pair, unit
    ):
        return self._linear_collection_efficiency_body(
            params,
            output.data,
            radii.data,
            is_first_in_pair.indicator.data,
            radii.idx.data,
            len(is_first_in_pair),
            unit,
        )
