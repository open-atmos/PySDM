"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state import State


class SDM:
    def __init__(self, kernel, dt, dv, n_sd, n_cell, backend):
        self.backend = backend

        self.dt = dt
        self.dv = dv
        self.kernel = kernel
        self.temp = backend.array(n_sd, dtype=float)
        self.rand = backend.array(n_sd // 2, dtype=float)
        self.prob = backend.array(n_sd, dtype=float)

        self.is_first_in_pair = backend.array(n_sd, dtype=int)  # TODO bool
        self.cell_start = backend.array(n_cell + 1, dtype=int)

    def __call__(self, state: State):
        assert state.is_healthy()

        self.toss_pairs(self.backend, self.is_first_in_pair, self.cell_start, state)

        self.compute_probability(self.backend, self.kernel, self.dt, self.dv, state, self.prob, self.temp,
                                 self.is_first_in_pair, self.cell_start)

        self.backend.urand(self.rand)
        self.compute_gamma(self.backend, self.prob, self.rand)

        self.coalescence(self.backend, state, gamma=self.prob)

        state.housekeeping()

    # TODO remove
    @staticmethod
    def compute_gamma(backend, prob, rand):
        backend.compute_gamma(prob, rand)

    @staticmethod
    def compute_probability(backend, kernel, dt, dv, state: State, prob, temp, is_first_in_pair, cell_start):

        kernel_temp = temp
        kernel(backend, kernel_temp, is_first_in_pair, state)

        backend.max_pair(prob, state.n, is_first_in_pair, state.idx, state.SD_num)
        backend.multiply(prob, kernel_temp)

        norm_factor = temp
        backend.normalize(prob, state.cell_id, cell_start, norm_factor, dt / dv)

    @staticmethod
    def toss_pairs(backend, is_first_in_pair, cell_start, state: State):
        state.unsort()
        state.sort_by_cell_id()
        backend.find_pairs(cell_start, is_first_in_pair, state.cell_id, state.idx, state.SD_num)

    @staticmethod
    def coalescence(backend, state: State, gamma):
        backend.coalescence(n=state.n,
                            idx=state.idx,
                            length=state.SD_num,
                            intensive=state.get_intensive_attrs(),
                            extensive=state.get_extensive_attrs(),
                            gamma=gamma,
                            healthy=state.healthy)
