"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class SDM:
    def __init__(self, kernel, dt, dv, n_sd, backend):
        self.backend = backend

        self.dt = dt
        self.dv = dv
        self.kernel = kernel
        self.temp = backend.array(n_sd // 2, dtype=float)
        self.rand = backend.array(n_sd // 2, dtype=float)
        self.prob = backend.array(n_sd // 2, dtype=float)

    @staticmethod
    def compute_gamma(backend, prob, rand):
        backend.multiply(prob, -1.)
        backend.sum(prob, rand)
        backend.floor(prob)
        backend.multiply(prob, -1.)

    @staticmethod
    def compute_probability(backend, kernel, dt, dv, state, prob, temp):
        kernel(backend, temp, state)

        backend.max_pair(prob, state.n, state.idx, state.SD_num)
        backend.multiply(prob, temp)

        if state.SD_num < 2:
            norm_factor = 0
        else:
            norm_factor = dt / dv * state.SD_num * (state.SD_num - 1) / 2 / (state.SD_num // 2)
        backend.multiply(prob, norm_factor)

    @staticmethod
    def toss_pairs(state):
        state.unsort()

    @staticmethod
    def coalescence(backend, state, gamma):
        backend.coalescence(n=state.n,
                            idx=state.idx,
                            length=state.SD_num,
                            intensive=state.get_intensive_attrs(),
                            extensive=state.get_extensive_attrs(),
                            gamma=gamma,
                            healthy=state.healthy)

    def __call__(self, state):
        assert state.is_healthy()

        self.toss_pairs(state)

        self.compute_probability(self.backend, self.kernel, self.dt, self.dv, state, self.prob, self.temp)

        self.backend.urand(self.rand)
        self.compute_gamma(self.backend, self.prob, self.rand)

        self.coalescence(self.backend, state, gamma=self.prob)

        state.housekeeping()
