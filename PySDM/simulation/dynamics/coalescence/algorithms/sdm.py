"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.particles import Particles


class SDM:

    def __init__(self, particles: Particles, kernel, croupier=''):
        self.particles = particles

        self.kernel = kernel

        if croupier == '':
            pass
        elif croupier == 'Shima_serial':
            self.toss_pairs = lambda _, __: 0
        else:
            raise NotImplementedError()

        self.temp = particles.backend.array(particles.n_sd, dtype=float)
        self.rand = particles.backend.array(particles.n_sd // 2, dtype=float)
        self.prob = particles.backend.array(particles.n_sd, dtype=float)
        self.is_first_in_pair = particles.backend.array(particles.n_sd, dtype=int)  # TODO bool
        self.cell_start = particles.backend.array(particles.mesh.n_cell + 1, dtype=int)

    def __call__(self):
        assert self.particles.state.is_healthy()

        self.toss_pairs(self.is_first_in_pair, self.cell_start)

        self.compute_probability(self.prob, self.temp, self.is_first_in_pair, self.cell_start)

        self.particles.backend.urand(self.rand)
        self.compute_gamma(self.prob, self.rand)

        self.particles.coalescence(gamma=self.prob)

        self.particles.state.housekeeping()

    def compute_gamma(self, prob, rand):
        self.particles.backend.compute_gamma(prob, rand)
    # TODO remove

    def compute_probability(self, prob, temp, is_first_in_pair, cell_start):
        kernel_temp = temp
        self.kernel(self.particles, kernel_temp, is_first_in_pair)

        self.particles.max_pair(prob, is_first_in_pair)
        self.particles.backend.multiply(prob, kernel_temp)

        norm_factor = temp
        self.particles.normalize(prob, cell_start, norm_factor)

    def toss_pairs(self, is_first_in_pair, cell_start):
        self.particles.state.unsort()
        self.particles.state.sort_by_cell_id()
        self.particles.find_pairs(cell_start, is_first_in_pair)

