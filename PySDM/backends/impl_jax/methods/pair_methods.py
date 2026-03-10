"""
CPU implementation of pairwise operations backend methods
"""

from functools import cached_property

import jax

from PySDM.backends.impl_common.backend_methods import BackendMethods


class PairMethods(BackendMethods):

    @cached_property
    def _find_pairs_body(self):
        @jax.jit
        def body(*, cell_start, is_first_in_pair, cell_id, cell_idx, idx, length):
            raise NotImplementedError()

        return body

    # pylint: disable=too-many-arguments
    def find_pairs(self, cell_start, is_first_in_pair, cell_id, cell_idx, idx):
        # IMPLEMENT
        return

    @cached_property
    def _max_pair_body(self):
        @jax.jit
        def body(data_out, data_in, is_first_in_pair, idx, i):
            data_out.at[i // 2].set(
                jax.numpy.maximum(data_in[idx[i]], data_in[idx[i + 1]])
                * is_first_in_pair[i]
            )
            return data_out

        return body

    def max_pair(self, data_out, data_in, is_first_in_pair, idx):
        temp_data_out = jax.numpy.empty(data_out.shape)
        indices = jax.numpy.arange(len(idx))

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

        data_out.data = jax.numpy.sum(temp_data_out, axis=0)

    @cached_property
    def _sort_within_pair_by_attr_body(self):
        @jax.jit
        def body(idx, length, is_first_in_pair, attr):
            raise NotImplementedError()

        return body

    def sort_within_pair_by_attr(self, idx, is_first_in_pair, attr):
        # IMPLEMENT
        return

    @cached_property
    def _sum_pair_body(self):
        @jax.jit
        def body(data_out, data_in, is_first_in_pair, idx, i):
            data_out.at[i // 2].set(
                (data_in[idx[i]] + data_in[idx[i + 1]]) * is_first_in_pair[i]
            )
            return data_out

        return body

    def sum_pair(self, data_out, data_in, is_first_in_pair, idx):
        # temp_data_out = jax.numpy.zeros(data_out.shape)
        temp_data_out = jax.numpy.empty(data_out.shape)
        indices = jax.numpy.arange(len(idx))

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

        data_out.data = jax.numpy.sum(temp_data_out, axis=0)
