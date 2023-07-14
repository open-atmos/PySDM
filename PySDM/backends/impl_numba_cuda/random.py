"""
random number generator class for Numba backend
"""

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

import math
from ..impl_common.random_common import RandomCommon


class Random(RandomCommon):
    def __init__(self, size, seed):
        """
        Parameters:
           size (int) - number of RNG states to create
           seed(uint64) - starting seed for list of generators
        """
        super().__init__(size, seed)

        # Number  block per grid depends on output arrays size so it is defined in __call__
        self.threadsperblock = (128,)

        self.rng_states = create_xoroshiro128p_states(size, seed)

    @staticmethod
    @cuda.jit
    def _compute_random_array(storage, rng_states):
        thread_id = cuda.grid(1)
        if thread_id < storage.shape[0]:
           storage[thread_id] = xoroshiro128p_uniform_float32(rng_states, thread_id)

    def __call__(self, storage):
        blockspergrid = (math.ceil(storage.shape[0] / self.threadsperblock[0]),)
        Random._compute_random_array[blockspergrid, self.threadsperblock](storage.data, self.rng_states)


