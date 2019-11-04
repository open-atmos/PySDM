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
        self.rand = backend.array(n_sd//2, dtype=float)
        self.prob = backend.array(n_sd, dtype=float)

        self.is_first_in_pair = backend.array(n_sd, dtype=int)  # TODO bool
        self.cell_start = backend.array(n_cell+1, dtype=int)

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
        n_cell = len(cell_start)
        # TODO: use backend
        for i in range(n_cell - 1):
            SD_num = cell_start[i+1] - cell_start[i]
            if SD_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = dt / dv * SD_num * (SD_num - 1) / 2 / (SD_num // 2)
        # TODO: use backend
        for d in range(len(prob)):
            prob[d] *= norm_factor[state.cell_id[d]]
#        backend.multiply(prob, norm_factor)

    @staticmethod
    def toss_pairs(is_first_in_pair, cell_start, state: State):
        state.unsort()
        state.sort_by_cell_id()

        # TODO: use backend
        for i in reversed(range(state.SD_num)):
            cell_id = state.cell_id[state.idx[i]]
            cell_start[cell_id] = i
        cell_start[-1] = state.SD_num

        for i in range(state.SD_num - 1):
            is_first_in_pair[i] = (
                state.cell_id[state.idx[i]] == state.cell_id[state.idx[i+1]] and
                (i - cell_start[state.cell_id[state.idx[i]]]) % 2 == 0
            )

    @staticmethod
    def coalescence(backend, state: State, gamma):
        backend.coalescence(n=state.n,
                            idx=state.idx,
                            length=state.SD_num,
                            intensive=state.get_intensive_attrs(),
                            extensive=state.get_extensive_attrs(),
                            gamma=gamma,
                            healthy=state.healthy)

    def __call__(self, state: State):
        assert state.is_healthy()

        self.toss_pairs(self.is_first_in_pair, self.cell_start, state)

        self.compute_probability(self.backend, self.kernel, self.dt, self.dv, state, self.prob, self.temp,
                                 self.is_first_in_pair, self.cell_start)

        self.backend.urand(self.rand)
        self.compute_gamma(self.backend, self.prob, self.rand)

        self.coalescence(self.backend, state, gamma=self.prob)

        state.housekeeping()
