"""
Created at 07.06.2019
"""

from PySDM.builder import Builder


class Coalescence:

    def __init__(self, builder: Builder, kernel, seed=None, max_substeps=128):
        self.core = builder.core

        kernel.register(builder)
        self.kernel = kernel

        self.enable = True
        self.stats_steps = 0
        self.adaptive = False
        self.max_substeps = max_substeps
        self.subs = 1
        self.croupier = 'local'

        self.temp = self.core.IndexedStorage.empty(self.core.n_sd, dtype=float)
        self.pairs_rand = self.core.Storage.empty(self.core.n_sd + self.max_substeps, dtype=float)
        self.rand = self.core.Storage.empty(self.core.n_sd // 2, dtype=float)
        self.prob = self.core.IndexedStorage.empty(self.core.n_sd, dtype=float)
        self.is_first_in_pair = self.core.IndexedStorage.empty(self.core.n_sd, dtype=int)  # TODO bool
        self.rnd = self.core.Random(self.core.n_sd, seed)

    def __call__(self):
        if self.enable:
            self.pairs_rand.urand(self.rnd)
            self.rand.urand(self.rnd)

            self.toss_pairs(self.is_first_in_pair,
                            self.pairs_rand[:self.core.n_sd])
            self.compute_probability(self.prob, self.is_first_in_pair, self.subs)

            subs = 0
            msub = 1
            for s in range(self.subs):
                sub = self.coalescence(self.prob, self.rand, self.adaptive, self.subs)
                subs += sub
                msub = max(msub, sub)
                if s < self.subs-1:
                    self.toss_pairs(self.is_first_in_pair,
                                    self.core.backend.range(self.pairs_rand, start=s, stop=self.core.n_sd + s))
                    self.compute_probability(self.prob, self.is_first_in_pair, self.subs)

            self.stats_steps += self.subs
            if self.adaptive:
                self.subs = min(self.max_substeps, int(((subs/self.subs) + msub)/2))

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand)

    def coalescence(self, prob, rand, adaptive, subs):
        self.compute_gamma(prob, rand)
        return self.core.coalescence(gamma=prob, adaptive=adaptive, subs=subs, adaptive_memory=self.temp)

    def compute_probability(self, prob, is_first_in_pair, subs):
        self.kernel(self.temp, is_first_in_pair)
        prob.max_pair(self.core.state['n'], is_first_in_pair)
        prob *= self.temp

        norm_factor = self.temp
        self.core.normalize(prob, norm_factor, subs)

    def toss_pairs(self, is_first_in_pair, u01):
        self.core.state.sanitize()
        self.core.state.permutation(u01, self.croupier == 'local')
        is_first_in_pair.find_pairs(self.core.state.cell_start, self.core.state['cell id'])
