"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np


class SDM:
    def __init__(self, kernel, dt, dv):
        M = 0  # TODO dependency to state[]
        N = 1
        self.probability = lambda sd1, sd2, n_sd: \
            max(sd1[N], sd2[N]) * kernel(sd1[M], sd2[M]) * dt / dv * n_sd * (n_sd - 1) / 2 / (n_sd//2)

    def __call__(self, state):
        n_sd = len(state)

        assert np.amin(state.n) > 0
        if n_sd < 2:
            return

        # toss pairs
        state.unsort()

        # collide iterating over pairs
        rand = np.random.uniform(0, 1, n_sd // 2)

        prob_func = np.vectorize(lambda j: self.probability(state[2*int(j)], state[2*int(j)+1], n_sd))
        prob = np.fromfunction(prob_func, rand.shape, dtype=float)

        # prob = np.empty_like(rand)
        # for i in range(1, n_sd, 2):
        #     prob[i // 2] = self.probability(state[idx[i]], state[idx[i - 1]], n_sd)

        gamma = np.floor(prob) + np.where(rand < prob - np.floor(prob), 1, 0)

        # TODO ! no loops
        for i in range(1, n_sd, 2):
            state.collide(i, i - 1, gamma[i//2])
