"""
CPU implementation of pairwise operations backend methods
"""

import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf


class PairMethods(BackendMethods):
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def distance_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                data_out[i // 2] = np.abs(data_in[idx[i]] - data_in[idx[i + 1]])

    @staticmethod
    def distance_pair(data_out, data_in, is_first_in_pair, idx):
        return PairMethods.distance_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def find_pairs_body(
        *, cell_start, is_first_in_pair, cell_id, cell_idx, idx, length
    ):
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            is_in_same_cell = cell_id[idx[i]] == cell_id[idx[i + 1]]
            is_even_index = (i - cell_start[cell_idx[cell_id[idx[i]]]]) % 2 == 0
            is_first_in_pair[i] = is_in_same_cell and is_even_index
        is_first_in_pair[length - 1] = False

    @staticmethod
    def find_pairs(cell_start, is_first_in_pair, cell_id, cell_idx, idx):
        return PairMethods.find_pairs_body(
            cell_start=cell_start.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
            cell_id=cell_id.data,
            cell_idx=cell_idx.data,
            idx=idx.data,
            length=len(idx),
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def max_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                data_out[i // 2] = max(data_in[idx[i]], data_in[idx[i + 1]])

    @staticmethod
    def max_pair(data_out, data_in, is_first_in_pair, idx):
        return PairMethods.max_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def min_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in numba.prange(length):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                data_out[i // 2] = min(data_in[idx[i]], data_in[idx[i + 1]])

    @staticmethod
    def min_pair(data_out, data_in, is_first_in_pair, idx):
        return PairMethods.min_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def sort_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                if data_in[idx[i]] < data_in[idx[i + 1]]:
                    data_out[i], data_out[i + 1] = data_in[idx[i + 1]], data_in[idx[i]]
                else:
                    data_out[i], data_out[i + 1] = data_in[idx[i]], data_in[idx[i + 1]]

    @staticmethod
    def sort_pair(data_out, data_in, is_first_in_pair, idx):
        return PairMethods.sort_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def sort_within_pair_by_attr_body(idx, length, is_first_in_pair, attr):
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                if attr[idx[i]] < attr[idx[i + 1]]:
                    idx[i], idx[i + 1] = idx[i + 1], idx[i]

    @staticmethod
    def sort_within_pair_by_attr(idx, is_first_in_pair, attr):
        PairMethods.sort_within_pair_by_attr_body(
            idx.data, len(idx), is_first_in_pair.indicator.data, attr.data
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def sum_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in numba.prange(length):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                data_out[i // 2] = data_in[idx[i]] + data_in[idx[i + 1]]

    @staticmethod
    def sum_pair(data_out, data_in, is_first_in_pair, idx):
        return PairMethods.sum_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def multiply_pair_body(data_out, data_in, is_first_in_pair, idx, length):
        data_out[:] = 0
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                data_out[i // 2] = data_in[idx[i]] * data_in[idx[i + 1]]

    @staticmethod
    def multiply_pair(data_out, data_in, is_first_in_pair, idx):
        return PairMethods.multiply_pair_body(
            data_out.data,
            data_in.data,
            is_first_in_pair.indicator.data,
            idx.data,
            len(idx),
        )
