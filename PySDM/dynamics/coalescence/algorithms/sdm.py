"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.particles_builder import ParticlesBuilder
from PySDM.dynamics.coalescence.seeds.incrementation import Incrementation


class SDM:

    def __init__(self, particles_builder: ParticlesBuilder, kernel, seed=None):
        self.particles = particles_builder.particles

        kernel.register(particles_builder)
        self.kernel = kernel

        self.temp = self.particles.backend.array(self.particles.n_sd, dtype=float)
        self.rand = self.particles.backend.array(self.particles.n_sd // 2, dtype=float)
        self.prob = self.particles.backend.array(self.particles.n_sd, dtype=float)
        self.is_first_in_pair = self.particles.backend.array(self.particles.n_sd, dtype=int)  # TODO bool
        self.seed = seed or Incrementation()

    def __call__(self):
        self.particles.backend.urand(self.temp, self.seed())

        self.toss_pairs(self.is_first_in_pair, self.temp)

        self.compute_probability(self.prob, self.temp, self.is_first_in_pair)

        self.particles.backend.urand(self.rand, self.seed())
        self.compute_gamma(self.prob, self.rand)

        self.particles.coalescence(gamma=self.prob)

    def compute_gamma(self, prob, rand):
        self.particles.backend.compute_gamma(prob, rand)

    def compute_probability(self, prob, temp, is_first_in_pair):
        kernel_temp = temp
        self.kernel(kernel_temp, is_first_in_pair)
        self.particles.max_pair(prob, is_first_in_pair)
        self.particles.backend.multiply(prob, kernel_temp)

        norm_factor = temp
        self.particles.normalize(prob, norm_factor)

    def toss_pairs(self, is_first_in_pair, u01):
        self.particles.permute(u01)
        self.particles.find_pairs(self.particles.state.cell_start, is_first_in_pair)


