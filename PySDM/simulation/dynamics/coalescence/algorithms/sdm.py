"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.particles import Particles
from PySDM.simulation.dynamics.coalescence.croupiers import global_FisherYates, local_FisherYates


class SDM:

    def __init__(self, particles: Particles, kernel, croupier='global_FisherYates'):
        self.particles = particles

        self.kernel = kernel

        if croupier == 'global_FisherYates':
            self.croupier = global_FisherYates
        elif croupier == 'local_FisherYates':
            self.croupier = local_FisherYates
        else:
            raise NotImplementedError()

        self.temp = particles.backend.array(particles.n_sd, dtype=float)
        self.rand = particles.backend.array(particles.n_sd // 2, dtype=float)
        self.prob = particles.backend.array(particles.n_sd, dtype=float)
        self.is_first_in_pair = particles.backend.array(particles.n_sd, dtype=int)  # TODO bool
        self.cell_start = particles.backend.array(particles.mesh.n_cell + 1, dtype=int)

    def __call__(self):
        assert self.particles.state.is_healthy()

        self.particles.backend.urand(self.temp)
        self.toss_pairs(self.is_first_in_pair, self.cell_start, self.temp)

        self.compute_probability(self.prob, self.temp, self.is_first_in_pair, self.cell_start)

        self.particles.backend.urand(self.rand)
        self.compute_gamma(self.prob, self.rand)

        self.particles.coalescence(gamma=self.prob)

        self.particles.state.housekeeping()

    def compute_gamma(self, prob, rand):
        self.particles.backend.compute_gamma(prob, rand)

    def compute_probability(self, prob, temp, is_first_in_pair, cell_start):
        kernel_temp = temp
        self.kernel(self.particles, kernel_temp, is_first_in_pair)

        self.particles.max_pair(prob, is_first_in_pair)
        self.particles.backend.multiply(prob, kernel_temp)

        norm_factor = temp
        self.particles.normalize(prob, cell_start, norm_factor)

    def toss_pairs(self, is_first_in_pair, cell_start, u01):
        self.croupier(self.particles, cell_start, u01)
        self.particles.find_pairs(cell_start, is_first_in_pair)


