"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State


class SDM:

    def __init__(self, simulation, kernel):
        self.simulation = simulation

        self.kernel = kernel

        self.temp = simulation.backend.array(simulation.n_sd, dtype=float)
        self.rand = simulation.backend.array(simulation.n_sd // 2, dtype=float)
        self.prob = simulation.backend.array(simulation.n_sd, dtype=float)
        self.is_first_in_pair = simulation.backend.array(simulation.n_sd, dtype=int)  # TODO bool
        self.cell_start = simulation.backend.array(simulation.n_cell + 1, dtype=int)

    def __call__(self):
        assert self.simulation.state.is_healthy()

        self.toss_pairs(self.is_first_in_pair, self.cell_start)

        self.compute_probability(self.prob, self.temp, self.is_first_in_pair, self.cell_start)

        self.simulation.backend.urand(self.rand)
        self.compute_gamma(self.prob, self.rand)

        self.coalescence(gamma=self.prob)

        self.simulation.state.housekeeping()

    def compute_gamma(self, prob, rand):
        self.simulation.backend.compute_gamma(prob, rand)
    # TODO remove

    def compute_probability(self, prob, temp, is_first_in_pair, cell_start):
        kernel_temp = temp
        self.kernel(self.simulation.backend, kernel_temp, is_first_in_pair, self.simulation.state)

        self.simulation.backend.max_pair(prob, self.simulation.state.n, is_first_in_pair, self.simulation.state.idx,
                                         self.simulation.state.SD_num)
        self.simulation.backend.multiply(prob, kernel_temp)

        norm_factor = temp
        self.simulation.backend.normalize(prob, self.simulation.state.cell_id, cell_start, norm_factor,
                                          self.simulation.dt / self.simulation.dv)

    def toss_pairs(self, is_first_in_pair, cell_start):
        self.simulation.state.unsort()
        self.simulation.state.sort_by_cell_id()
        self.simulation.backend.find_pairs(cell_start, is_first_in_pair,
                                           self.simulation.state.cell_id,
                                           self.simulation.state.idx,
                                           self.simulation.state.SD_num)

    def coalescence(self, gamma):
        state = self.simulation.state
        self.simulation.backend.coalescence(n=state.n,
                                            idx=state.idx,
                                            length=state.SD_num,
                                            intensive=state.get_intensive_attrs(),
                                            extensive=state.get_extensive_attrs(),
                                            gamma=gamma,
                                            healthy=state.healthy)
