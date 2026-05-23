"""
random number generator class for JAX backend
"""

from jax import random

from ..impl_common.random_common import RandomCommon

#  TIP: can be called asynchronously
#  TIP: sometimes only half array is needed


class Random(RandomCommon):  # pylint: disable=too-few-public-methods
    def __init__(self, size, seed):
        super().__init__(size, seed)
        self.key = random.key(seed)

    def __call__(self, storage):
        storage.data.at[:].set(
            random.uniform(self.key, storage.shape, storage.dtype, 0, 1)
        )
