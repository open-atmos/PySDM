"""
JAX implementation of backend methods for particle collisions
"""

from functools import cached_property

import jax
import jax.numpy as jnp

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_jax.storage import Storage


def flag_zero_multiplicity(j, k, multiplicity, healthy):
    raise NotImplementedError()


class CollisionsMethods(BackendMethods):
    @cached_property
    def _collision_coalescence_body(self):
        @jax.jit
        def body(
            multiplicity,
            idx,
            attributes,
            gamma,
            is_first_in_pair,
        ):
            i = jnp.arange(len(multiplicity) // 2)
            offset = 1 - is_first_in_pair[2 * i]
            # offset = 0
            j = idx[2 * i + offset]
            k = idx[2 * i + 1 + offset]
            mj = multiplicity[j]
            mk = multiplicity[k]

            new_n = mj - jnp.int64(gamma[i]) * mk  # ?

            pos_mask = new_n > 0
            zero_mask = ~pos_mask

            # CASE new_n > 0
            mult_pos_updates = jnp.where(pos_mask, new_n, multiplicity[j])
            attr_pos_delta = gamma[i] * attributes[:, j]

            attr_pos_delta = attr_pos_delta * pos_mask

            # CASE new_n <= 0
            new_attr = gamma[i] * attributes[:, j] + attributes[:, k]

            new_attr = new_attr * zero_mask

            mult_j = jnp.where(
                pos_mask,
                mult_pos_updates,
                jnp.where(zero_mask, mk // 2, multiplicity[j]),
            )
            mult_k = jnp.where(
                zero_mask,
                jnp.where(zero_mask, mk - (mk // 2), multiplicity[k]),
                multiplicity[k],
            )

            multiplicity = multiplicity.at[j].set(mult_j)

            multiplicity = multiplicity.at[k].set(mult_k)

            attributes = attributes.at[:, k].add(attr_pos_delta)

            attributes = attributes.at[:, j].set(
                jnp.where(zero_mask, new_attr, attributes[:, j])
            )
            attributes = attributes.at[:, k].set(
                jnp.where(zero_mask, new_attr, attributes[:, k])
            )

            return multiplicity, attributes

        return body

    def collision_coalescence(
        self,
        *,
        multiplicity,
        idx,
        attributes,
        gamma,
        healthy,  # pylint: disable=unused-argument
        cell_id,  # pylint: disable=unused-argument
        coalescence_rate,  # pylint: disable=unused-argument
        is_first_in_pair,
        # TODO #1913: implement rates, multi-cell collisions and healthy
    ):
        multiplicity.data, attributes.data = self._collision_coalescence_body(
            multiplicity.data,
            idx.data,
            attributes.data,
            gamma.data,
            is_first_in_pair.indicator.data,
        )
        multiplicity.data.block_until_ready()

    @cached_property
    def _compute_gamma_body(self):
        def body(prob, rand, idx, multiplicity, is_first_in_pair, i):
            out = jnp.ceil(prob[i] - rand[i])
            offset = 1 - is_first_in_pair[2 * i]
            j = idx[2 * i + offset]
            k = idx[2 * i + 1 + offset]

            prop = multiplicity[j] // multiplicity[k]
            return jnp.minimum(out.astype(int), prop)

        return jax.jit(jax.vmap(body, (None, None, None, None, None, 0)))

    def compute_gamma(
        self,
        *,
        prob,
        rand,
        multiplicity,
        cell_id,  # pylint: disable=unused-argument
        collision_rate_deficit,  # pylint: disable=unused-argument
        collision_rate,  # pylint: disable=unused-argument
        is_first_in_pair,
        out,
        # TODO #1913: implement rates, multi-cell collisions and healthy
    ):
        indices = jnp.arange(len(multiplicity) // 2)
        out.data = self._compute_gamma_body(
            prob.data,
            rand.data,
            multiplicity.idx.data,
            multiplicity.data,
            is_first_in_pair.indicator.data,
            indices,
        ).block_until_ready()

    @staticmethod
    def make_cell_caretaker(
        idx_shape, idx_dtype, cell_start_len, scheme="default"
    ):  # pylint: disable=unused-argument
        # cell_start_len present for API compliance
        class CellCaretaker:  # pylint: disable=too-few-public-methods
            def __init__(self, idx_shape, idx_dtype, scheme):
                assert scheme == "default"
                self.tmp_idx = Storage.empty(idx_shape, idx_dtype)

            def __call__(self, cell_id, cell_idx, cell_start, idx):
                length = len(idx)
                cell_start.data = (
                    CollisionsMethods._counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data,
                        idx.data,
                        cell_id.data,
                        cell_idx.data,
                        length,
                        cell_start.data,
                    )
                )

        return CellCaretaker(idx_shape, idx_dtype, scheme)

    @cached_property
    def _normalize_body(self):
        @jax.jit
        def body(prob, cell_start, timestep, dv):
            sd_num = cell_start[1] - cell_start[0]

            def loop_body(i, prob):
                norm_factor = (sd_num >= 2) * (
                    timestep / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
                )
                prob = prob.at[i].multiply(norm_factor)
                return prob

            return jax.lax.fori_loop(0, prob.shape[0], loop_body, prob)

        return body

    # pylint: disable=too-many-arguments
    def normalize(self, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
        prob.data = self._normalize_body(
            prob.data, cell_start.data, timestep, dv
        ).block_until_ready()

    @staticmethod
    # pylint: disable=too-many-arguments
    def _counting_sort_by_cell_id_and_update_cell_start(
        new_idx,
        idx,
        cell_id,
        cell_idx,
        length,
        cell_start,  # pylint: disable=unused-argument
    ):
        # TODO #1913:  implement sorting
        cell_start = cell_start.at[0].set(0)
        return cell_start
