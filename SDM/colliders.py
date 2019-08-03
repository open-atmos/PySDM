"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.backends.default import Default


class SDM:
    def __init__(self, kernel, dt, dv, n_sd, backend=Default):
        self.backend = backend

        self.dt = dt
        self.dv = dv
        self.kernel = kernel
        self.ker = backend.array(n_sd // 2, dtype=float)
        self.rand = backend.array(n_sd // 2, dtype=float)
        self.prob = backend.array(n_sd // 2, dtype=float)

    # TODO
    @staticmethod
    def compute_gamma(backend, prob, rand):
        prob[:] = -prob
        backend.sum(prob, rand)
        backend.floor(prob)
        prob[:] = -prob

    def __call__(self, state):
        assert state.is_healthy()

        # toss pairs
        state.unsort()

        # TODO (segments)
        # state.sort_by('z', stable=True)  # state.stable_sort_by_segment()

        # collide iterating over pairs
        self.backend.urand(self.rand)

        # kernel
        self.kernel(self.backend, self.ker, state)

        # probability, explain
        self.backend.max_pair(self.prob, state.n, state.idx, state.SD_num)
        self.backend.multiply(self.prob, self.ker)
        # TODO segment
        if state.SD_num < 2:
            norm_factor = 0
        else:
            norm_factor = self.dt / self.dv * state.SD_num * (state.SD_num - 1) / 2 / (state.SD_num // 2)
        self.backend.multiply(self.prob, norm_factor)

        self.compute_gamma(self.backend, self.prob, self.rand)

        # TODO (potential optimisation... some doubts...)
        # state.sort_by_pairs('n')

        # TODO (when an example with intensive param will be available)
        # self.backend.intesive_attr_coalescence(data=state.get_intensive(), gamma=self.gamma)

        for attrs in state.get_extensive_attrs().values():
            self.backend.extensive_attr_coalescence(n=state.n,
                                                    idx=state.idx,
                                                    length=state.SD_num,
                                                    data=attrs,
                                                    gamma=self.prob)

        self.backend.n_coalescence(n=state.n, idx=state.idx, length=state.SD_num, gamma=self.prob)

        state.housekeeping()
