"""
CPU implementation of pairwise operations backend methods
"""

from functools import cached_property

import jax
import jax.numpy as jnp

from PySDM.backends.impl_common.backend_methods import BackendMethods


class PairMethods(BackendMethods):

    @cached_property
    def _find_pairs_body(self):
        @jax.jit
        def body(cell_start, cell_id, cell_idx, idx, i):

            is_in_same_cell = cell_id[idx[i]] == cell_id[idx[i + 1]]
            is_even_index = (i - cell_start[cell_idx[cell_id[idx[i]]]]) % 2 == 0
            is_first_in_pair = is_in_same_cell & is_even_index

            return is_first_in_pair

        return body

    # pylint: disable=too-many-arguments
    def find_pairs(self, cell_start, is_first_in_pair, cell_id, cell_idx, idx):

        indices = jnp.arange(len(idx) - 1)

        mapped_find_pairs = jax.vmap(
            self._find_pairs_body,
            (None, None, None, None, 0),
        )
        is_first_in_pair.indicator.data = jnp.append(
            mapped_find_pairs(
                cell_start.data,
                cell_id.data,
                cell_idx.data,
                idx.data,
                indices,
            ),
            jnp.array([False]),
        )

    @cached_property
    def _max_pair_body(self):
        @jax.jit
        def body(data_out, data_in, is_first_in_pair, idx, i):
            data_out.at[i // 2].set(
                jnp.maximum(data_in[idx[i]], data_in[idx[i + 1]]) * is_first_in_pair[i]
            )
            return data_out

        return body

    def max_pair(self, data_out, data_in, is_first_in_pair, idx):
        temp_data_out = jnp.empty(data_out.shape)
        indices = jnp.arange(len(idx))

        mapped_max_pair = jax.vmap(
            self._max_pair_body,
            (None, None, None, None, 0),
        )
        temp_data_out = mapped_max_pair(
            temp_data_out,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            indices,
        )

        data_out.data = jnp.sum(temp_data_out, axis=0)

    @cached_property
    def _sort_within_pair_by_attr_body(self):
        # @jax.jit
        # def body(idx, attr, i, is_first_in_pair):
        #     j = i + 1
        #     h = i - 1
        #     idx_i = idx[i]
        #     idx_j = idx[j]
        #     idx_h = idx[h]

        #     inverse_j = (attr[idx_i] < attr[idx_j]) & is_first_in_pair[i]
        #     inverse_h = (attr[idx_h] < attr[idx_i]) & is_first_in_pair[h]
        #     # idx = idx.at[i].set(idx_j * inverse + idx_i * (not inverse))
        #     # idx = idx.at[i + 1].set(idx_i * inverse + idx_j * (not inverse))

        #     return [idx_j * inverse + idx_i * ~inverse,
        #             idx_i * inverse + idx_j * ~inverse]
        # return [inverse, inverse]

        @jax.jit
        def body(is_first_in_pair, attr, idx):
            should_swap = is_first_in_pair & (attr[idx] < jnp.roll(attr[idx], -1))
            second_to_swap = jnp.roll(should_swap, 1)

            swapped_idx = jnp.where(should_swap, jnp.roll(idx, -1), idx)
            swapped_idx = jnp.where(second_to_swap, jnp.roll(idx, 1), swapped_idx)

            return swapped_idx

        body

        return body

    def sort_within_pair_by_attr(self, idx, is_first_in_pair, attr):

        idx.data = self._sort_within_pair_by_attr_body(
            is_first_in_pair.indicator.data, attr.data, idx.data
        )
        # indices = [i for i, val in enumerate(is_first_in_pair.indicator.data) if val]
        # indices = jnp.array(indices)
        # print(f"{indices=}")

        # # padding

        # mapped_sort_within_pair_by_attr = jax.vmap(
        #     self._sort_within_pair_by_attr_body,
        #     (None, None, 0),
        # )
        # # print(f"{indices=}")
        # pairs = mapped_sort_within_pair_by_attr(
        #     idx.data, attr.data, indices
        # )
        # print(f"{pairs=}")
        # stacked_pairs = jnp.stack(pairs, 1)
        # temp = idx.data[-1]
        #     # stacked_pairs = jnp.append(stacked_pairs, jnp.array(idx.data[-1]))
        # print(f"{stacked_pairs=}")
        # idx.data = jnp.concatenate(stacked_pairs, 0)
        # if len(idx) % 2 != 0:
        #     idx.data = jnp.append(idx.data, temp)

    @cached_property
    def _sum_pair_body(self):
        @jax.jit
        def body(data_out, data_in, is_first_in_pair, idx, i):
            data_out = data_out.at[i // 2].set(
                (data_in[idx[i]] + data_in[idx[i + 1]]) * is_first_in_pair[i]
            )
            return data_out

        return body

    def sum_pair(self, data_out, data_in, is_first_in_pair, idx):
        # temp_data_out = jnp.zeros(data_out.shape)
        temp_data_out = jnp.empty(data_out.shape)
        indices = jnp.arange(len(idx) - 1)

        mapped_sum_pair = jax.vmap(
            self._sum_pair_body,
            (None, None, None, None, 0),
        )
        temp_data_out = mapped_sum_pair(
            temp_data_out,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            indices,
        )

        data_out.data = jnp.sum(temp_data_out, axis=0)
