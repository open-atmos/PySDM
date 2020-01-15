"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.particles import Particles


class SDM:

    def __init__(self, particles: Particles, kernel):
        self.particles = particles

        self.kernel = kernel

        self.temp = particles.backend.array(particles.n_sd, dtype=float)
        self.rand = particles.backend.array(particles.n_sd // 2, dtype=float)
        self.prob = particles.backend.array(particles.n_sd, dtype=float)
        self.is_first_in_pair = particles.backend.array(particles.n_sd, dtype=int)  # TODO bool

    def __call__(self):
        assert self.particles.state.is_healthy()

        self.particles.backend.urand(self.temp)
        self.toss_pairs(self.is_first_in_pair, self.temp)

        self.compute_probability(self.prob, self.temp, self.is_first_in_pair)

        self.particles.backend.urand(self.rand)
        self.compute_gamma(self.prob, self.rand)

        self.particles.coalescence(gamma=self.prob)

        self.particles.state.housekeeping()

    def compute_gamma(self, prob, rand):
        self.particles.backend.compute_gamma(prob, rand)

    def compute_probability(self, prob, temp, is_first_in_pair):
        kernel_temp = temp
        self.kernel(self.particles, kernel_temp, is_first_in_pair)

        self.particles.max_pair(prob, is_first_in_pair)
        self.particles.backend.multiply(prob, kernel_temp)

        norm_factor = temp
        self.particles.normalize(prob, norm_factor)

    def toss_pairs(self, is_first_in_pair, u01):
        self.particles.permute(u01)
        self.particles.find_pairs(self.particles.state.cell_start, is_first_in_pair)


