"""
CPU implementation of backend methods for particle collisions
"""

from functools import cached_property
import numpy as np

import jax
import jax.numpy as jnp

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_jax.storage import Storage


def pair_indices(i, idx, is_first_in_pair, prob_like):
    assert False
    offset = 1 - is_first_in_pair[2 * i]
    j = idx[2 * i + offset]
    k = idx[2 * i + 1 + offset]

    return j, k, False


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
            mj_new = mk // 2
            mk_new = mk - mj_new

            mult_j_zero = jnp.where(zero_mask, mj_new, multiplicity[j])
            mult_k_zero = jnp.where(zero_mask, mk_new, multiplicity[k])

            new_attr = gamma[i] * attributes[:, j] + attributes[:, k]

            new_attr = new_attr * zero_mask

            mult_j = jnp.where(pos_mask, mult_pos_updates, mult_j_zero)
            mult_k = jnp.where(zero_mask, mult_k_zero, multiplicity[k])

            multiplicity = multiplicity.at[j].set(mult_j)

            multiplicity = multiplicity.at[k].set(mult_k)

            # return mult_j, mult_k

            # update attr for new_n > 0
            attributes = attributes.at[:, k].add(attr_pos_delta)
            # attributes = attributes.at[0, k].add(attr_pos_delta)
            # attributes = attributes.at[1, k].add(attr_pos_delta)

            # update attr for new_n == 0
            attributes = attributes.at[:, j].set(
                jnp.where(zero_mask, new_attr, attributes[:, j])
            )
            attributes = attributes.at[:, k].set(
                jnp.where(zero_mask, new_attr, attributes[:, k])
            )

            # attributes = attributes.at[0, j].set(
            #     jnp.where(zero_mask, new_attr, attributes[0, j])
            # )
            # attributes = attributes.at[0, k].set(
            #     jnp.where(zero_mask, new_attr, attributes[0, k])
            # )
            # attributes = attributes.at[1, j].set(
            #     jnp.where(zero_mask, new_attr, attributes[1, j])
            # )
            # attributes = attributes.at[1, k].set(
            #     jnp.where(zero_mask, new_attr, attributes[1, k])
            # )
            # # return multiplicity

            # if new_n > 0:
            #     multiplicity[j] = new_n
            #     for a in range(len(attributes)):
            #         attributes[a, k] += gamma[i] * attributes[a, j]
            # else:  # new_n == 0
            #     multiplicity[j] = multiplicity[k] // 2
            #     multiplicity[k] = multiplicity[k] - multiplicity[j]
            #     for a in range(len(attributes)):
            #         attributes[a, j] = gamma[i] * attributes[a, j] + attributes[a, k]
            #         attributes[a, k] = attributes[a, j]

            # offset = 0
            # j = idx[2 * 0 + offset]
            # k = idx[2 * 0 + 1 + offset]
            # print(f"{j=}, {k=}")
            # print(f"{idx=}")
            # print(f"{multiplicity=}")
            # print(f"{is_first_in_pair=}")

            # def loop_body(i, multiplicity_attributes):
            #     mult, attr = multiplicity_attributes

            #     # offset = 1 - is_first_in_pair[2 * i]
            #     offset = 0
            #     # j = idx[2 * i + offset]
            #     # k = idx[2 * i + 1 + offset]
            #     j = 0
            #     k = 1
            #     new_n = mult[j] - jnp.int64(gamma[i]) * mult[k]
            #     a = jnp.arange(len(attr))
            #     def n_above_0(i, _multiplicity_attributes):
            #         _mult, _attr = _multiplicity_attributes
            #         _mult = _mult.at[j].set(new_n)
            #         _attr = _attr.at[a, k].add(gamma[i] * _attr[a, j])

            #         return (_mult, _attr)

            #     def n_equal_0(i, _multiplicity_attributes):
            #         _mult, _attr = _multiplicity_attributes
            #         temp = _mult[k] // 2
            #         _mult = _mult.at[j].set(temp).at[k].set(_mult[k] - temp)
            #         # _mult = _mult.at[j].set(1).at[k].set(2)

            #         # _mult = _mult.at[0].set(1).at[1].set(1)
            #         # _mult = _mult.at[k].set(_mult[k] - _mult[j])
            #         _attr = _attr.at[a, j].set(gamma[i] * _attr[a, j] + _attr[a, k])
            #         _attr = _attr.at[a, k].set(_attr[a, j])

            #         return (_mult, _attr)

            #     return jax.lax.cond(
            #         new_n > 0,
            #         n_above_0,
            #         n_equal_0,
            #         i, multiplicity_attributes
            #     )

            # return jax.lax.fori_loop(0, len(multiplicity // 2), loop_body, (multiplicity, attributes))

            return multiplicity, attributes

            # return multiplicity, attributes

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
        # pass
        # indices = jnp.arange(len(multiplicity) // 2)
        # mapped_collision_coalescence = jax.vmap(
        #     self._collision_coalescence_body, (None, None, None, None, None, 0)
        # )
        # multiplicity.data, attributes.data = mapped_collision_coalescence(
        #     multiplicity.data,
        #     idx.data,
        #     attributes.data,
        #     gamma.data,
        #     is_first_in_pair.indicator.data,
        #     indices,
        # )

        # print(f"pre-coalescence: {multiplicity.data=}")
        # print(f"pre-coalescence pairs: {is_first_in_pair.indicator.data}")
        # breakpoint()
        multiplicity.data, attributes.data = self._collision_coalescence_body(
            multiplicity.data,
            idx.data,
            attributes.data,
            gamma.data,
            is_first_in_pair.indicator.data,
        )
        multiplicity.data.block_until_ready()
        # print(f"post-coalescence: {multiplicity.data=}")

        # print(f"{mult_pos_updates=} \n {mult_j_zero=} \n {mult_k_zero=} \n {mult_k=} \n {pos_mask=}")

    @cached_property
    def _compute_gamma_body(self):
        # pylint: disable=too-many-arguments,too-many-locals
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
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
        out,
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
    def make_cell_caretaker(idx_shape, idx_dtype, cell_start_len, scheme="default"):
        class CellCaretaker:
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

    # @cached_property
    # def _normalize_body(self):
    #     # pylint: disable=too-many-arguments
    #     def body(prob, cell_start, timestep, dv, i):
    #         sd_num = cell_start[1] - cell_start[0]
    #         # Fixed this with (sd_num >= 2), need to add the n_cell loop, and then the normalization pass
    #         norm_factor = (sd_num >= 2) * (
    #             timestep / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
    #         )
    #         prob = prob.at[i].set(norm_factor)
    #         return prob

    #     return jax.jit(jax.vmap(body, (None, None, None, None, 0)))

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
        new_idx, idx, cell_id, cell_idx, length, cell_start
    ):
        # TODO #1913:  implement sorting
        cell_start = cell_start.at[0].set(0)
        return cell_start
