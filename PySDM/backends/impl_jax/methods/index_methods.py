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
            # def loop_body(i, idx):
            #     j = jax.numpy.int32(i + (u01[i] * (length - i)))
            #     temp_i = idx[i]
            #     temp_j = idx[j]
            #     idx = idx.at[i].set(temp_j)
            #     idx = idx.at[j].set(temp_i)
            #     return idx
            idx = idx.at[:].set(jax.numpy.argsort(u01, stable=False))
            # return jax.lax.fori_loop(0, length-1, loop_body, idx)
            return idx

        return body

    def shuffle_global(self, idx, u01):
        u01.permute(idx)
        # idx.data = self._shuffle_global_body(
        #     idx.data,
        #     u01,
        #     idx.length,
        # ).block_until_ready()

    def shuffle_local(self, idx, u01, cell_start):
        # TODO: implement shuffle_local
        self.shuffle_global(idx, u01).block_until_ready()
