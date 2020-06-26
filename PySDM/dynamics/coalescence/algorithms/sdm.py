"""


Created at 07.06.2019
"""

from PySDM.particles_builder import ParticlesBuilder
from PySDM.dynamics.coalescence.seeds.incrementation import Incrementation


class SDM:

    def __init__(self, particles_builder: ParticlesBuilder, kernel, seed=None, max_substeps=128):
        self.particles = particles_builder.particles

        kernel.register(particles_builder)
        self.kernel = kernel

        self.enable = True
        self.stats_steps = 0
        self.adaptive = False
        self.max_substeps = max_substeps
        self.subs = 1
        self.seed = seed or Incrementation(1)

        self.temp = self.particles.backend.array(self.particles.n_sd, dtype=float)
        self.pairs_rand = self.particles.backend.array(self.particles.n_sd + self.max_substeps, dtype=float)
        self.rand = self.particles.backend.array(self.particles.n_sd // 2, dtype=float)
        self.prob = self.particles.backend.array(self.particles.n_sd, dtype=float)
        self.is_first_in_pair = self.particles.backend.array(self.particles.n_sd, dtype=int)  # TODO bool

    def __call__(self):
        if self.enable:
            self.particles.backend.urand(self.pairs_rand, self.seed())
            self.particles.backend.urand(self.rand, self.seed())

            self.toss_pairs(self.is_first_in_pair,
                            self.particles.backend.range(self.pairs_rand, stop=self.particles.n_sd))
            self.compute_probability(self.prob, self.is_first_in_pair, self.subs)

            subs = 0
            msub = 1
            for s in range(self.subs):
                sub = self.coalescence(self.prob, self.rand, self.adaptive, self.subs)
                subs += sub
                msub = max(msub, sub)
                if s < self.subs-1:
                    self.toss_pairs(self.is_first_in_pair,
                                    self.particles.backend.range(self.pairs_rand, start=s, stop=self.particles.n_sd+s))
                    self.compute_probability(self.prob, self.is_first_in_pair, self.subs)

            self.stats_steps += self.subs
            if self.adaptive:
                self.subs = min(self.max_substeps, int(((subs/self.subs) + msub)/2))

    def compute_gamma(self, prob, rand):
        self.particles.backend.compute_gamma(prob, rand)

    def coalescence(self, prob, rand, adaptive, subs):
        self.compute_gamma(prob, rand)
        return self.particles.coalescence(gamma=prob, adaptive=adaptive, subs=subs, adaptive_memory=self.temp)

    def compute_probability(self, prob, is_first_in_pair, subs):
        self.kernel(self.temp, is_first_in_pair)
        self.particles.max_pair(prob, is_first_in_pair)
        self.particles.backend.multiply(prob, self.temp)

        norm_factor = self.temp
        self.particles.normalize(prob, norm_factor, subs)

    def toss_pairs(self, is_first_in_pair, u01):
        self.particles.state.sanitize()
        self.particles.permute(u01)
        self.particles.find_pairs(self.particles.state.cell_start, is_first_in_pair)
