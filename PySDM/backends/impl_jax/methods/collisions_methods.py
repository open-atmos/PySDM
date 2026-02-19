"""
CPU implementation of backend methods for particle collisions
"""

from functools import cached_property
import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_jax.storage import Storage


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


class CollisionsMethods(BackendMethods):
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
            # TODO #1731 - shared docstring for all backends
            for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
                out[i] = np.ceil(prob[i] - rand[i])
                j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, out)
                if skip_pair:
                    continue
                prop = multiplicity[j] // multiplicity[k]
                g = min(int(out[i]), prop)
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
