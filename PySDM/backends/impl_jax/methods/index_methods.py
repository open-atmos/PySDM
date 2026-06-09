"""
CPU implementation of shuffling and sorting backend methods
"""

from functools import cached_property

import jax

from PySDM.backends.impl_common.backend_methods import BackendMethods


class IndexMethods(BackendMethods):

    @cached_property
    def shuffle_global(self):
        def body(idx, u01):
            idx.data = idx.data.at[:].set(jax.numpy.argsort(u01))

        return body

    def shuffle_local(self, idx, u01, cell_start):
        # TODO: implement shuffle_local
        self.shuffle_global(idx, u01)
