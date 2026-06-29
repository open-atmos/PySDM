"""
CPU implementation of pairwise operations backend methods
"""

from functools import cached_property, partial

import jax
import jax.numpy as jnp

from PySDM.backends.impl_common.backend_methods import BackendMethods


class PairMethods(BackendMethods):

    @cached_property
    def _find_pairs_body(self):  
        @partial(jax.jit, static_argnames=['size'])
        def body(is_first_in_pair, cell_start, cell_id, cell_idx, idx, size):

            indices = jnp.arange(size)
            idx_roll = jnp.roll(idx, 1)
            is_in_same_cell = cell_id[idx] == cell_id[idx_roll]
            is_first_in_pair = (indices - cell_start[cell_idx[cell_id[idx]]]) % 2 == 0

            is_first_in_pair = (is_in_same_cell & is_first_in_pair).at[-1].set(False)
            return is_first_in_pair

        return body
    
    # @cached_property
    # def _find_pairs_body(self):
    #     @jax.jit
    #     def body(is_first_in_pair, cell_start, cell_id, cell_idx, idx):
    #         def loop_body(i, is_first_in_pair):
    #             is_in_same_cell = cell_id[idx[i]] == cell_id[idx[i + 1]]
    #             is_even_index = (i - cell_start[cell_idx[cell_id[idx[i]]]]) % 2 == 0
    #             is_first_in_pair = is_first_in_pair.at[i].set(is_in_same_cell & is_even_index)
    #             return is_first_in_pair
    #         is_first_in_pair = jax.lax.fori_loop(0, len(idx - 1), loop_body, is_first_in_pair).at[-1].set(False)
    #         return is_first_in_pair
    #     return body

    # pylint: disable=too-many-arguments
    def find_pairs(self, cell_start, is_first_in_pair, cell_id, cell_idx, idx):

        # print(f"{cell_start.data=}, {cell_id.data=}, {cell_idx.data=}, {idx.data=}")

        is_first_in_pair.indicator.data = self._find_pairs_body(
                is_first_in_pair.indicator.data,
                cell_start.data,
                cell_id.data,
                cell_idx.data,
                idx.data,
                len(idx.data),
            ).block_until_ready()
        # assert is_first_in_pair.indicator.data[0]

    @cached_property
    def _max_pair_body(self):
        @jax.jit  # TODO: test with jit
        def body(data_out, data_in, is_first_in_pair, idx):
            def loop_body(i, data_out):
                def max_pair(i, data_out):
                    # data_out = data_out.at[i//2].set(
                    #     ((data_in[idx[i]] + data_in[idx[i + 1]]) +
                    #      jnp.abs(data_in[idx[i]] - data_in[idx[i + 1]]))/2
                    #     )
                    # data_out = data_out.at[i//2].set(data_in[idx[i]] + data_in[idx[i + 1]])
                    data_out = data_out.at[i // 2].set(
                        jnp.maximum(data_in[idx[i]], data_in[idx[i + 1]])
                    )
                    return data_out

                return jax.lax.cond(
                    is_first_in_pair[i],
                    max_pair,
                    lambda _, data_out: data_out,
                    i,
                    data_out,
                )

            return jax.lax.fori_loop(0, len(idx), loop_body, data_out)

        return body

    def max_pair(self, data_out, data_in, is_first_in_pair, idx):

        data_out.data = self._max_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
        ).block_until_ready()

    @cached_property
    def _sort_within_pair_by_attr_body(self):

        @jax.jit
        def body(is_first_in_pair, attr, idx):
            should_swap = is_first_in_pair & (attr[idx] < jnp.roll(attr[idx], -1))
            second_to_swap = jnp.roll(should_swap, 1)

            swapped_idx = jnp.where(should_swap, jnp.roll(idx, -1), idx)
            swapped_idx = jnp.where(second_to_swap, jnp.roll(idx, 1), swapped_idx)

            return swapped_idx

        # body

        return body

    def sort_within_pair_by_attr(self, idx, is_first_in_pair, attr):

        idx.data = self._sort_within_pair_by_attr_body(
            is_first_in_pair.indicator.data, attr.data, idx.data
        ).block_until_ready()

    @cached_property
    def _sum_pair_body(self):
        @jax.jit
        def body(data_out, data_in, is_first_in_pair, idx):
            data_out = data_out.at[:].set(0)  # ?? might slow it down

            def loop_body(i, data_out):
                def sum_pair(i, data_out):
                    data_out = data_out.at[i // 2].set(
                        data_in[idx[i]] + data_in[idx[i + 1]]
                    )
                    return data_out

                return jax.lax.cond(
                    is_first_in_pair[i],
                    sum_pair,
                    lambda _, data_out: data_out,
                    i,
                    data_out,
                )

            return jax.lax.fori_loop(0, len(idx), loop_body, data_out)

        return body

    def sum_pair(self, data_out, data_in, is_first_in_pair, idx):
        data_out.data = self._sum_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
        ).block_until_ready()
