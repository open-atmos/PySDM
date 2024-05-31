"""
CPU implementation of pairwise operations backend methods
"""

from functools import cached_property

import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods


class PairMethods(BackendMethods):
    @cached_property
    def _distance_pair_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(data_out, data_in, is_first_in_pair, idx, length):
            data_out[:] = 0
            for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
                if is_first_in_pair[i]:
                    data_out[i // 2] = np.abs(data_in[idx[i]] - data_in[idx[i + 1]])

        return body

    def distance_pair(self, data_out, data_in, is_first_in_pair, idx):
        return self._distance_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @cached_property
    def _find_pairs_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(*, cell_start, is_first_in_pair, cell_id, cell_idx, idx, length):
            for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
                is_in_same_cell = cell_id[idx[i]] == cell_id[idx[i + 1]]
                is_even_index = (i - cell_start[cell_idx[cell_id[idx[i]]]]) % 2 == 0
                is_first_in_pair[i] = is_in_same_cell and is_even_index
            is_first_in_pair[length - 1] = False

        return body

    # pylint: disable=too-many-arguments
    def find_pairs(self, cell_start, is_first_in_pair, cell_id, cell_idx, idx):
        return self._find_pairs_body(
            cell_start=cell_start.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
            cell_id=cell_id.data,
            cell_idx=cell_idx.data,
            idx=idx.data,
            length=len(idx),
        )

    @cached_property
    def _max_pair_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(data_out, data_in, is_first_in_pair, idx, length):
            data_out[:] = 0
            for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
                if is_first_in_pair[i]:
                    data_out[i // 2] = max(data_in[idx[i]], data_in[idx[i + 1]])

        return body

    def max_pair(self, data_out, data_in, is_first_in_pair, idx):
        return self._max_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @cached_property
    def _min_pair_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(data_out, data_in, is_first_in_pair, idx, length):
            data_out[:] = 0
            for i in numba.prange(length):  # pylint: disable=not-an-iterable
                if is_first_in_pair[i]:
                    data_out[i // 2] = min(data_in[idx[i]], data_in[idx[i + 1]])

        return body

    def min_pair(self, data_out, data_in, is_first_in_pair, idx):
        return self._min_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @cached_property
    def _sort_pair_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(data_out, data_in, is_first_in_pair, idx, length):
            data_out[:] = 0
            for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
                if is_first_in_pair[i]:
                    if data_in[idx[i]] < data_in[idx[i + 1]]:
                        data_out[i], data_out[i + 1] = (
                            data_in[idx[i + 1]],
                            data_in[idx[i]],
                        )
                    else:
                        data_out[i], data_out[i + 1] = (
                            data_in[idx[i]],
                            data_in[idx[i + 1]],
                        )

        return body

    def sort_pair(self, data_out, data_in, is_first_in_pair, idx):
        return self._sort_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @cached_property
    def _sort_within_pair_by_attr_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(idx, length, is_first_in_pair, attr):
            for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
                if is_first_in_pair[i]:
                    if attr[idx[i]] < attr[idx[i + 1]]:
                        idx[i], idx[i + 1] = idx[i + 1], idx[i]

        return body

    def sort_within_pair_by_attr(self, idx, is_first_in_pair, attr):
        self._sort_within_pair_by_attr_body(
            idx.data, len(idx), is_first_in_pair.indicator.data, attr.data
        )

    @cached_property
    def _sum_pair_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(data_out, data_in, is_first_in_pair, idx, length):
            data_out[:] = 0
            for i in numba.prange(length):  # pylint: disable=not-an-iterable
                if is_first_in_pair[i]:
                    data_out[i // 2] = data_in[idx[i]] + data_in[idx[i + 1]]

        return body

    def sum_pair(self, data_out, data_in, is_first_in_pair, idx):
        return self._sum_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @cached_property
    def _multiply_pair_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(data_out, data_in, is_first_in_pair, idx, length):
            data_out[:] = 0
            for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
                if is_first_in_pair[i]:
                    data_out[i // 2] = data_in[idx[i]] * data_in[idx[i + 1]]

        return body

    def multiply_pair(self, data_out, data_in, is_first_in_pair, idx):
        return self._multiply_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )
