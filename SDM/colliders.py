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
        self.temp = backend.array(n_sd // 2, dtype=float)
        self.rand = backend.array(n_sd // 2, dtype=float)
        self.prob = backend.array(n_sd // 2, dtype=float)

    # TODO
    @staticmethod
    def compute_gamma(backend, prob, rand):
        prob[:] = -prob
        backend.sum(prob, rand)
        backend.floor(prob)
        prob[:] = -prob

    @staticmethod
    def compute_probability(backend, kernel, dt, dv, state, prob, temp):
        # kernel
        kernel(backend, temp, state)

        backend.max_pair(prob, state.n, state.idx, state.SD_num)
        backend.multiply(prob, temp)
        # TODO segment
        if state.SD_num < 2:
            norm_factor = 0
        else:
            norm_factor = dt / dv * state.SD_num * (state.SD_num - 1) / 2 / (state.SD_num // 2)
        backend.multiply(prob, norm_factor)

    def __call__(self, state):
        assert state.is_healthy()

        # toss pairs
        state.unsort()

        # TODO (segments)
        # state.sort_by('z', stable=True)  # state.stable_sort_by_segment()

        self.compute_probability(self.backend, self.kernel, self.dt, self.dv, state, self.prob, self.temp)

        self.backend.urand(self.rand)
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
