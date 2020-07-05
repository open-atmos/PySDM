"""
Created at 07.06.2019
"""

from PySDM.particles_builder import ParticlesBuilder


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

        self.temp = self.particles.backend.IndexedStorage.empty(self.particles.n_sd, dtype=float)
        self.pairs_rand = self.particles.backend.Storage.empty(self.particles.n_sd + self.max_substeps, dtype=float)
        self.rand = self.particles.backend.Storage.empty(self.particles.n_sd // 2, dtype=float)
        self.prob = self.particles.backend.IndexedStorage.empty(self.particles.n_sd, dtype=float)
        self.is_first_in_pair = self.particles.backend.IndexedStorage.empty(self.particles.n_sd, dtype=int)  # TODO bool
        self.rnd = self.particles.backend.Random(self.particles.n_sd, seed)

        self.enable = True

    def __call__(self):
        if self.enable:
            self.pairs_rand.urand(self.rnd)
            self.rand.urand(self.rnd)

            self.toss_pairs(self.is_first_in_pair,
                            self.pairs_rand[:self.particles.n_sd])
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
        prob *= self.temp

        norm_factor = self.temp
        self.particles.normalize(prob, norm_factor, subs)

    def toss_pairs(self, is_first_in_pair, u01):
        self.particles.state.sanitize()
        self.particles.permute(u01)
        self.particles.find_pairs(is_first_in_pair)
