"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from SDM.backends.numpy import Numpy as backend


class SDM:
    def __init__(self, kernel, dt, dv, n_sd):
        self.probability = lambda sd1n, sd2n, sd1x, sd2x, n_sd: \
            max(sd1n, sd2n) * kernel(sd1x, sd2x) * dt / dv * n_sd * (n_sd - 1) / 2 / (n_sd // 2)
        self.rand = backend.array((n_sd // 2,), type=float)
        self.prob = backend.array((n_sd // 2,), float)
        self.gamma = backend.array((n_sd // 2,), float)

    def __call__old(self, state):
        n_sd = state.SD_num

        assert backend.amin(state.n) > 0

        # toss pairs
        state.unsort()

        # TODO stable sort by segments

        # collide iterating over pairs
        backend.urand(self.rand)

        backend.transform(self.prob, lambda j: self.probability(state.n[0][2 * j],
                                                                state.n[0][(2 * j) + 1],
                                                                state['x'][2 * j],
                                                                state['x'][(2 * j) + 1],
                                                                n_sd))

        backend.transform(self.gamma, lambda j: self.prob[j] // 1 + (self.rand[j] < self.prob[j] - self.prob[j] // 1))

        collide = self.Collide(state, self.gamma)
        backend.foreach(state.n, collide)

    # TODO intensive/extensive
    class Collide:
        def __init__(self, state, gamma):
            self.state = state
            self.gamma = gamma

        def __call__(self, i):
            # TODO in segments
            if i % 2 == 0: return
            j = i // 2 + 1
            k = j - 1

            gamma = self.gamma[j // 2]

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

    def __call__(self, state):
        assert state.is_healthy()

        # toss pairs
        state.unsort()

        # TODO
        # state.sort_by('z', stable=True)  # state.stable_sort_by_segment()

        # collide iterating over pairs
        backend.urand(self.rand)

        backend.transform(self.prob, lambda j: self.probability(state._n[state._idx[2 * j]],
                                                                state._n[state._idx[2 * j + 1]],
                                                                state._x[state._idx[2 * j]],
                                                                state._x[state._idx[2 * j + 1]],
                                                                state.SD_num),
                          state.SD_num // 2)

        backend.transform(self.gamma, lambda j: self.prob[j] // 1 + (self.rand[j] < self.prob[j] - self.prob[j] // 1),
                          state.SD_num // 2)

        # state.sort_by_pairs('n')

        # backend.intesive_attr_coalescence(data=state.get_intensive(), gamma=self.gamma)
        for attrs in state.get_extensive_attrs().values():
            backend.extensive_attr_coalescence(n=state._n,
                                               idx=state._idx,
                                               length=state.SD_num,
                                               data=attrs,
                                               gamma=self.gamma)
        backend.n_coalescence(n=state._n, idx=state._idx, length=state.SD_num, gamma=self.gamma)

        state.housekeeping()
