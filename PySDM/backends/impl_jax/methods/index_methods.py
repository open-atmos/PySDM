"""
CPU implementation of shuffling and sorting backend methods
"""

from functools import cached_property

import jax
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods


class IndexMethods(BackendMethods):

    @cached_property
    def _shuffle_global_body(self):
        @jax.jit
        def body(idx, u01, length):
            idx = idx.at[:].set(jax.numpy.argsort(u01, stable=False))
            return idx

        return body

    def shuffle_global(self, idx, u01):
        # TODO #1913: decide whether to use u01 argsort or random.permute
        u01.permute(idx)

    def shuffle_local(self, idx, u01, cell_start):
        # TODO #1913: implement shuffle_local

        job = self.shuffle_global(idx, u01)
        if self.block_until_ready:
            job.block_until_ready()
