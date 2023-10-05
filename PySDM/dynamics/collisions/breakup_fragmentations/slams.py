"""
Based on [Jokulsdottir & Archer 2016 (GMD)](https://doi.org/10.5194/gmd-9-1455-2016)
for ocean particles
"""


class SLAMS:
    def __init__(self, mass_min=0.0, nfmax=None):
        self.particulator = None
        self.p_vec = None
        self.sum_of_masses = None
        self.mass_min = mass_min
        self.nfmax = nfmax

    def __call__(self, nf, frag_mass, u01, is_first_in_pair):
        self.sum_of_masses.sum(
            self.particulator.attributes["water mass"], is_first_in_pair
        )
        self.particulator.backend.slams_fragmentation(
            n_fragment=nf,
            frag_mass=frag_mass,
            x_plus_y=self.sum_of_masses,
            probs=self.p_vec,
            rand=u01,
            mass_min=self.mass_min,
            nfmax=self.nfmax,
        )

    def register(self, builder):
        self.particulator = builder.particulator
        self.p_vec = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.sum_of_masses = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
