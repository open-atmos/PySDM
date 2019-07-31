"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.backends.numpy import Numpy as backend


class SDM:
    def __init__(self, kernel, dt, dv, n_sd):
        self.dt = dt
        self.dv = dv
        self.kernel = kernel
        self.ker = backend.array((n_sd // 2,), type=float)
        self.rand = backend.array((n_sd // 2,), type=float)
        self.prob = backend.array((n_sd // 2,), type=float)

    # TODO
    @staticmethod
    def compute_gamma(backend_TODO, prob: backend.storage, rand: backend.storage):
        prob[:] = -prob
        backend_TODO.sum(prob, rand)
        backend_TODO.floor(prob)
        prob[:] = -prob

    def __call__(self, state):
        assert state.is_healthy()

        # toss pairs
        state.unsort()

        # TODO (segments)
        # state.sort_by('z', stable=True)  # state.stable_sort_by_segment()

        # collide iterating over pairs
        backend.urand(self.rand)

        # kernel
        self.kernel(backend, self.ker, state)

        # probability, explain
        backend.max_pair(self.prob, state._n, state._idx, state.SD_num)
        backend.multiply(self.prob, self.ker)
        # TODO segment
        if state.SD_num < 2:
            norm_factor = 0
        else:
            norm_factor = self.dt / self.dv * state.SD_num * (state.SD_num - 1) / 2 / (state.SD_num // 2)
        backend.multiply(self.prob, norm_factor)

        self.compute_gamma(backend, self.prob, self.rand)

        # TODO (potential optimisation... some doubts...)
        # state.sort_by_pairs('n')

        # TODO (when an example with intensive param will be available)
        # backend.intesive_attr_coalescence(data=state.get_intensive(), gamma=self.gamma)

        for attrs in state.get_extensive_attrs().values():
            backend.extensive_attr_coalescence(n=state._n,
                                               idx=state._idx,
                                               length=state.SD_num,
                                               data=attrs,
                                               gamma=self.prob)

        backend.n_coalescence(n=state._n, idx=state._idx, length=state.SD_num, gamma=self.prob)

        state.housekeeping()
