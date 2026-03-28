"""
CPU implementation of backend methods for particle collisions
"""

from functools import cached_property
import numpy as np

import jax

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_jax.storage import Storage


def pair_indices(i, idx, is_first_in_pair, prob_like):
    raise NotImplementedError()


def flag_zero_multiplicity(j, k, multiplicity, healthy):
    raise NotImplementedError()


def coalesce(  # pylint: disable=too-many-arguments
    i, j, k, cid, multiplicity, gamma, attributes, coalescence_rate
):
    raise NotImplementedError()


class CollisionsMethods(BackendMethods):
    @cached_property
    def _collision_coalescence_body(self):
        @jax.jit
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
            raise NotImplementedError()

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
        raise NotImplementedError()

    @cached_property
    def _compute_gamma_body(self):
        # pylint: disable=too-many-arguments,too-many-locals
        @jax.jit
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
            raise NotImplementedError()

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
        raise NotImplementedError()

    @staticmethod
    def make_cell_caretaker(idx_shape, idx_dtype, cell_start_len, scheme="default"):
        class DummyCaretaker:
            def __call__(self, *args, **kwds):
                return

        return DummyCaretaker()

    @cached_property
    def _normalize_body(self):
        # pylint: disable=too-many-arguments
        def body(prob, cell_start, timestep, dv, i):
            sd_num = cell_start[1] - cell_start[0]
            norm_factor = timestep / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
            prob.at[i].set(norm_factor)
            return prob

        return body

    # pylint: disable=too-many-arguments
    def normalize(self, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
        indices = jax.numpy.arange(prob.shape[0])
        temp_prob = jax.numpy.empty(prob.shape)

        normalize_func = jax.vmap(self._normalize_body, (None, None, None, None, 0))

        temp_prob = normalize_func(temp_prob, cell_start.data, timestep, dv, indices)

        prob.data = jax.numpy.sum(temp_prob, axis=0)

    @staticmethod
    # pylint: disable=too-many-arguments
    def _counting_sort_by_cell_id_and_update_cell_start(
        new_idx, idx, cell_id, cell_idx, length, cell_start
    ):
        raise NotImplementedError()
