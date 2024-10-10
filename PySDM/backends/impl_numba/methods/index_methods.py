"""
CPU implementation of shuffling and sorting backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class IndexMethods(BackendMethods):
    @cached_property
    def identity_index(self):
        @numba.njit(**self.default_jit_flags)
        def body(idx):
            for i in numba.prange(len(idx)):  # pylint: disable=not-an-iterable
                idx[i] = i

        return body

    @cached_property
    def shuffle_global(self):
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def body(idx, length, u01):
            for i in range(length - 1, 0, -1):
                j = int(u01[i] * (i + 1))
                idx[i], idx[j] = idx[j], idx[i]

        return body

    @cached_property
    def shuffle_local(self):
        @numba.njit(**self.default_jit_flags)
        def body(idx, u01, cell_start):
            # pylint: disable=not-an-iterable
            for c in numba.prange(len(cell_start) - 1):
                for i in range(cell_start[c + 1] - 1, cell_start[c], -1):
                    j = int(
                        cell_start[c] + u01[i] * (cell_start[c + 1] - cell_start[c])
                    )
                    idx[i], idx[j] = idx[j], idx[i]

        return body

    @staticmethod
    def sort_by_key(idx, attr):
        idx.data[:] = attr.data.argsort(kind="stable")[::-1]
