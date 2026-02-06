"""
CPU implementation of shuffling and sorting backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class IndexMethods(BackendMethods):

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
