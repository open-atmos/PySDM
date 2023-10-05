"""
P(x) = exp(-x / lambda); specified in mass units
"""
# TODO #796: introduce common code with Feingold fragmentation, including possible limiter


class Exponential:
    def __init__(self, scale, mass_min=0.0, nfmax=None):
        self.particulator = None
        self.scale = scale
        self.mass_min = mass_min
        self.nfmax = nfmax
        self.sum_of_masses = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.sum_of_masses = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )

    def __call__(self, nf, frag_mass, u01, is_first_in_pair):
        self.sum_of_masses.sum(
            self.particulator.attributes["water mass"], is_first_in_pair
        )
        self.particulator.backend.exp_fragmentation(
            n_fragment=nf,
            scale=self.scale,
            frag_mass=frag_mass,
            x_plus_y=self.sum_of_masses,
            rand=u01,
            mass_min=self.mass_min,
            nfmax=self.nfmax,
        )
