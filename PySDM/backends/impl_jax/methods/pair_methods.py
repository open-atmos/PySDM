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
        raise NotImplementedError()

    @cached_property
    def _max_pair_body(self):
        @jax.jit
        def body(data_out, data_in, is_first_in_pair, idx, length):
            raise NotImplementedError()

        return body

    def max_pair(self, data_out, data_in, is_first_in_pair, idx):
        raise NotImplementedError()

    @cached_property
    def _sort_within_pair_by_attr_body(self):
        @jax.jit
        def body(idx, length, is_first_in_pair, attr):
            raise NotImplementedError()

        return body

    def sort_within_pair_by_attr(self, idx, is_first_in_pair, attr):
        raise NotImplementedError()

    @cached_property
    def _sum_pair_body(self):
        @jax.jit
        def body(data_out, data_in, is_first_in_pair, idx, length):
            raise NotImplementedError()

        return body

    def sum_pair(self, data_out, data_in, is_first_in_pair, idx):
        raise NotImplementedError()
