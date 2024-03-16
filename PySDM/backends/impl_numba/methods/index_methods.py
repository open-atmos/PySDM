"""
CPU implementation of shuffling and sorting backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf


class IndexMethods(BackendMethods):
    @cached_property
    def identity_index(self):
        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def body(idx):
            for i in numba.prange(len(idx)):  # pylint: disable=not-an-iterable
                idx[i] = i

        return body

    @cached_property
    def shuffle_global(self):
        @numba.njit(
            **{**conf.JIT_FLAGS, "parallel": False, "fastmath": self.formulae.fastmath}
        )
        def body(idx, length, u01):
            for i in range(length - 1, 0, -1):
                j = int(u01[i] * (i + 1))
                idx[i], idx[j] = idx[j], idx[i]

        return body

    @cached_property
    def shuffle_local(self):
        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def body(idx, u01, cell_start):
            for c in numba.prange(
                len(cell_start) - 1
            ):  # pylint: disable=not-an-iterable
                for i in range(cell_start[c + 1] - 1, cell_start[c], -1):
                    j = int(
                        cell_start[c] + u01[i] * (cell_start[c + 1] - cell_start[c])
                    )
                    idx[i], idx[j] = idx[j], idx[i]

        return body

    @staticmethod
    def sort_by_key(idx, attr):
        idx.data[:] = attr.data.argsort(kind="stable")[::-1]
