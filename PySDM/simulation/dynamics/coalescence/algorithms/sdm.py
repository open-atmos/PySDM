"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

class SDM:

    def __init__(self, particles, kernel):
        self.particles = particles

        self.kernel = kernel

        self.temp = particles.backend.array(particles.n_sd, dtype=float)
        self.rand = particles.backend.array(particles.n_sd // 2, dtype=float)
        self.prob = particles.backend.array(particles.n_sd, dtype=float)
        self.is_first_in_pair = particles.backend.array(particles.n_sd, dtype=int)  # TODO bool
        self.cell_start = particles.backend.array(particles.environment.n_cell + 1, dtype=int)

    def __call__(self):
        assert self.particles.state.is_healthy()

        self.toss_pairs(self.is_first_in_pair, self.cell_start)

        self.compute_probability(self.prob, self.temp, self.is_first_in_pair, self.cell_start)

        self.particles.backend.urand(self.rand)
        self.compute_gamma(self.prob, self.rand)

        self.coalescence(gamma=self.prob)

        self.particles.state.housekeeping()

    def compute_gamma(self, prob, rand):
        self.particles.backend.compute_gamma(prob, rand)
    # TODO remove

    def compute_probability(self, prob, temp, is_first_in_pair, cell_start):
        kernel_temp = temp
        self.kernel(self.particles.backend, kernel_temp, is_first_in_pair, self.particles.state)

        self.particles.backend.max_pair(prob, self.particles.state.n, is_first_in_pair, self.particles.state.idx,
                                        self.particles.state.SD_num)
        self.particles.backend.multiply(prob, kernel_temp)

        norm_factor = temp
        self.particles.backend.normalize(prob, self.particles.state.cell_id, cell_start, norm_factor,
                                         self.particles.environment.dt / self.particles.environment.dv)

    def toss_pairs(self, is_first_in_pair, cell_start):
        self.particles.state.unsort()
        self.particles.state.sort_by_cell_id()
        self.particles.backend.find_pairs(cell_start, is_first_in_pair,
                                          self.particles.state.cell_id,
                                          self.particles.state.idx,
                                          self.particles.state.SD_num)

    def coalescence(self, gamma):
        state = self.particles.state
        self.particles.backend.coalescence(n=state.n,
                                           idx=state.idx,
                                           length=state.SD_num,
                                           intensive=state.get_intensive_attrs(),
                                           extensive=state.get_extensive_attrs(),
                                           gamma=gamma,
                                           healthy=state.healthy)
