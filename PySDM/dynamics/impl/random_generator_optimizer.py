"""
TODO: class to be removed
"""


class RandomGeneratorThing:  # pylint: disable=too-many-instance-attributes
    def __init__(self, dt_min, seed):
        self.particulator = None
        self.dt_min = dt_min
        self.seed = seed
        self.pairs_rand = None
        self.rand = None
        self.rnd = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.pairs_rand = self.particulator.Storage.empty(
            self.particulator.n_sd, dtype=float
        )
        self.rand = self.particulator.Storage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.rnd = self.particulator.Random(self.particulator.n_sd, self.seed)

    def get_random_arrays(self):
        self.pairs_rand.urand(self.rnd)
        self.rand.urand(self.rnd)
        return self.pairs_rand[0 : self.particulator.n_sd], self.rand
