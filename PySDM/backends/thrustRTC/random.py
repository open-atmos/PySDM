"""
Created at 25.08.2020
"""

import numpy as np
from .conf import trtc, rndrtc
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS


#  TIP: sometimes only half array is needed

class Random:
    __urand_init_rng_state_body = trtc.For(['rng', 'states', 'seed'], 'i', '''
        rng.state_init(1234, i, 0, states[i]);
        ''')

    __urand_body = trtc.For(['states', 'vec_rnd'], 'i', '''
        vec_rnd[i]=states[i].rand01();
        ''')

    def __init__(self, size, seed=None):
        rng = rndrtc.DVRNG()
        self.generator = trtc.device_vector('RNGState', size)
        self.size = size
        seed = seed or np.random.randint(0, 2*16)
        dseed = trtc.DVInt64(seed)
        Random.__urand_init_rng_state_body.launch_n(size, [rng, self.generator, dseed])

    @nice_thrust(**NICE_THRUST_FLAGS)
    def __call__(self, storage):
        assert len(storage) <= self.size
        Random.__urand_body.launch_n(len(storage), [self.generator, storage.data])
