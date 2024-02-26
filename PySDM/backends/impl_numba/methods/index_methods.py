"""
CPU implementation of shuffling and sorting backend methods
"""

import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def draw_random_int(start: int, end: int, u01: float):
    return min(int(start + u01 * (end - start + 1)), end)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def fisher_yates_shuffle(idx, u01, start, end, random_offset=0):
    for i in range(end - 1, start, -1):
        j = draw_random_int(start=start, end=i, u01=u01[random_offset + i])
        idx[i], idx[j] = idx[j], idx[i]


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def merge(
    idx, u01, start, middle, end, random_offset
):  # pylint: disable=too-many-arguments
    i = start
    j = middle

    while True:
        if u01[random_offset + i] > 0.5:
            if j == end:
                break
            idx[i], idx[j] = idx[j], idx[i]
            j += 1
        else:
            if i == j:
                break
        i += 1

    fisher_yates_shuffle(idx, u01, i, end, random_offset)


class IndexMethods(BackendMethods):
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def identity_index(idx):
        for i in numba.prange(len(idx)):  # pylint: disable=not-an-iterable
            idx[i] = i

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def shuffle_global(idx, length, u01, cutoff=None):
        depth = 0
        while cutoff is not None and (length >> depth) > cutoff:
            depth += 1

        split_start = np.linspace(0, length, (1 << depth) + 1).astype(np.uint32)

        for c in numba.prange(len(split_start) - 1):  # pylint: disable=not-an-iterable
            fisher_yates_shuffle(idx, u01, split_start[c], split_start[c + 1])

        for i in range(1, depth + 1):
            for c in numba.prange(1 << (depth - i)):  # pylint: disable=not-an-iterable
                start = c * i * 2
                merge(
                    idx,
                    u01,
                    split_start[start],
                    split_start[start + i],
                    split_start[start + i + i],
                    length * i,
                )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def shuffle_local(idx, u01, cell_start):
        for c in numba.prange(len(cell_start) - 1):  # pylint: disable=not-an-iterable
            fisher_yates_shuffle(idx, u01, cell_start[c], cell_start[c + 1])

    @staticmethod
    def sort_by_key(idx, attr):
        idx.data[:] = attr.data.argsort(kind="stable")[::-1]
