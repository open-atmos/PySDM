"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from SDM.backends.numpy import Numpy as backend


class SDM:
    def __init__(self, kernel, dt, dv):
        M = 0  # TODO dependency to state!!!
        N = 1
        self.probability = lambda sd1, sd2, n_sd: \
            max(sd1[N], sd2[N]) * kernel(sd1[M], sd2[M]) * dt / dv * n_sd * (n_sd - 1) / 2 / (n_sd//2)



    def __call__(self, state):
        n_sd = state.SD_num

        assert np.amin(state['n']) > 0

        # toss pairs
        state.unsort()

        # collide iterating over pairs
        rand = backend.array((n_sd // 2,), type=float)
        backend.urand(rand)

        prob = backend.array(rand.shape, float)
        backend.transform(prob, lambda j: self.probability(state.get_SD(2*j), state.get_SD((2*j)+1), n_sd))

        gamma = backend.array(rand.shape, float)
        backend.transform(gamma, lambda j: prob[j]//1 + (rand[j] < prob[j] - prob[j]//1))

        collide = self.Collide(state, gamma)
        backend.foreach(state['n'], collide)

    # TODO intensive/extensive
    class Collide:
        def __init__(self, state, gamma):
            self.state = state
            self.gamma = gamma

        def __call__(self, i):
            if i % 2 == 0: return
            j = i//2 + 1
            k = j - 1

            gamma = self.gamma[j//2]

            if self.state['n'][j] < self.state['n'][k]:
                j, k = k, j

            gamma = min(gamma, self.state['n'][j] // self.state['n'][k])

            if self.state['n'][k] != 0:  # TODO: guaranteed by undertaker
                n = self.state['n'][j] - gamma * self.state['n'][k]
                if n > 0:
                    self.state['n'][j] = n
                    self.state['x'][k] += gamma * self.state['x'][j]
                else:  # n == 0
                    self.state['n'][j] = self.state['n'][k] // 2
                    self.state['n'][k] = self.state['n'][k] - self.state['n'][j]
                    self.state['x'][j] = gamma * self.state['x'][j] + self.state['x'][k]
                    self.state['x'][k] = self.state['x'][j]


# nowe atrybuty (inty): n, idx(prm), sgmnt(grid),
    def __call__2(self, state):
        # toss pairs
        state.unsort()
        state.stable_sort_by_segment()


        # collide iterating over pairs
        rand = np.random.uniform(0, 1, state.SD_num // 2)

        prob_func = np.vectorize(lambda j: self.probability(state.get_SD(2*int(j)), state.get_SD((2*int(j))+1), n_sd))
        prob = np.fromfunction(prob_func, rand.shape, dtype=float)

        gamma = np.floor(prob) + np.where(rand < prob - np.floor(prob), 1, 0)

        # TODO ! no loops
        for i in range(1, n_sd, 2):
            state.collide(i, i - 1, gamma[i//2])

        state.housekeeping()
