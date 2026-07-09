"""
random number generator class for JAX backend
"""

from functools import cached_property, partial

from jax import random
import jax

from ..impl_common.random_common import RandomCommon

#  TIP: can be called asynchronously
#  TIP: sometimes only half array is needed


class Random(RandomCommon):  # pylint: disable=too-few-public-methods
    JAX = "TEMPORARY FLAG"

    def __init__(self, size, seed):
        super().__init__(size, seed)
        self.key = random.key(seed)

    @cached_property
    def _call_body(self):
        @partial(jax.jit, static_argnames=["shape", "dtype"])
        def body(data, subkey, shape, dtype):
            data = data.at[:].set(random.uniform(subkey, shape, dtype, 0, 1))
            return data

        return body

    def u01(self, storage):
        new_key, subkey = random.split(self.key)

        storage.data = self._call_body(
            storage.data,
            subkey,
            storage.shape,
            storage.dtype,
        ).block_until_ready()

        self.key = new_key

    @cached_property
    def _permute_body(self):
        @jax.jit
        def body(storage, key):
            storage = storage.at[:].set(jax.random.permutation(key, storage))
            return storage

        return body

    def permute(self, storage):
        new_key, subkey = random.split(self.key)
        storage.data = self._permute_body(
            storage.data,
            subkey,
        )
        self.key = new_key
