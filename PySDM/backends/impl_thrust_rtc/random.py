"""
random number generator class for ThrustRTC backend (using CURandRTC)
"""

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..impl_common.random_common import RandomCommon
from .conf import rndrtc, trtc

#  TIP: sometimes only half array is needed


class Random(RandomCommon):  # pylint: disable=too-few-public-methods
    __urand_init_rng_state_body = trtc.For(
        ["rng", "states", "seed"],
        "i",
        """
        rng.state_init(seed, i, 0, states[i]);
        """,
    )

    __urand_body = trtc.For(
        ["states", "vec_rnd"],
        "i",
        """
        vec_rnd[i] = states[i].rand01();
        """,
    )

    def __init__(self, size, seed):
        super().__init__(size, seed)
        rng = rndrtc.DVRNG()
        self.generator = trtc.device_vector("RNGState", size)
        dseed = trtc.DVInt64(seed)
        Random.__urand_init_rng_state_body.launch_n(size, [rng, self.generator, dseed])

    @nice_thrust(**NICE_THRUST_FLAGS)
    def __call__(self, storage):
        assert len(storage) <= self.size
        Random.__urand_body.launch_n(len(storage), [self.generator, storage.data])
