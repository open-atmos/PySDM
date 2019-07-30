"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.backends.numpy import Numpy as backend


class SDM:
    def __init__(self, kernel, dt, dv, n_sd):
        self.probability = lambda sd1n, sd2n, sd1x, sd2x, n_sd: \
            max(sd1n, sd2n) * kernel(sd1x, sd2x) * dt / dv * n_sd * (n_sd - 1) / 2 / (n_sd // 2)
        self.rand = backend.array((n_sd // 2,), type=float)
        self.prob = backend.array((n_sd // 2,), float)
        self.gamma = backend.array((n_sd // 2,), float)

    def __call__(self, state):
        assert state.is_healthy()

        # toss pairs
        state.unsort()

        # TODO (segments)
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

        # TODO (potential optimisation... some doubts...)
        # state.sort_by_pairs('n')

        # TODO (when an example with intensive param will be available)
        # backend.intesive_attr_coalescence(data=state.get_intensive(), gamma=self.gamma)

        for attrs in state.get_extensive_attrs().values():
            backend.extensive_attr_coalescence(n=state._n,
                                               idx=state._idx,
                                               length=state.SD_num,
                                               data=attrs,
                                               gamma=self.gamma)

        backend.n_coalescence(n=state._n, idx=state._idx, length=state.SD_num, gamma=self.gamma)

        state.housekeeping()
