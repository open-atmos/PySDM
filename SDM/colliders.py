"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class SDM:
    def __init__(self, kernel, dt, dv):
        M = 0  # TODO dependency to state!!!
        N = 1
        self.probability = lambda sd1, sd2, n_sd: \
            max(sd1[N], sd2[N]) * kernel(sd1[M], sd2[M]) * dt / dv * n_sd * (n_sd - 1) / 2 / (n_sd//2)

    def __call__(self, state):
        n_sd = state.SD_num

        assert np.amin(state['n']) > 0
        if n_sd < 2:
            return

        # toss pairs
        state.unsort()

        # collide iterating over pairs
        rand = np.random.uniform(0, 1, n_sd // 2)

        prob_func = np.vectorize(lambda j: self.probability(state.get_SD(2*int(j)), state.get_SD((2*int(j))+1), n_sd))
        prob = np.fromfunction(prob_func, rand.shape, dtype=float)

        gamma = np.floor(prob) + np.where(rand < prob - np.floor(prob), 1, 0)

        # TODO ! no loops
        for i in range(1, n_sd, 2):
            state.collide(i, i - 1, gamma[i//2])
