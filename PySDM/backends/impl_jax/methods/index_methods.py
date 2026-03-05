"""
CPU implementation of shuffling and sorting backend methods
"""

from functools import cached_property

import jax

from PySDM.backends.impl_common.backend_methods import BackendMethods


class IndexMethods(BackendMethods):

    @cached_property
    def shuffle_local(self):
        @jax.jit
        def body(idx, u01, cell_start):
            raise NotImplementedError()

        return body
