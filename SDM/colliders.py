"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np


class SDM:
    def __init__(self, kernel, dt, dv):
        self.probability = lambda m1, m2, n_sd: \
            kernel(m1, m2) * dt / dv * n_sd * (n_sd - 1) / 2 / (n_sd//2)

    def __call__(self, state):
        assert np.amin(state.n) > 0
        n_sd = len(state)

        # toss pairs
        idx = np.random.permutation(np.arange(n_sd))

        # collide iterating over pairs
        rand = np.random.uniform(0, 1, n_sd // 2)
        prob = np.empty_like(rand)

        # TODO ! no loops
        for i in range(1, n_sd, 2):
            prob[i // 2] = self.probability(state[idx[i]], state[idx[i - 1]], n_sd)

        gamma = np.floor(prob) + np.where(rand < prob - np.floor(prob), 1, 0)

        # TODO ! no loops
        for i in range(1, n_sd, 2):
            state.collide(idx[i], idx[i - 1], gamma[i - 1])
