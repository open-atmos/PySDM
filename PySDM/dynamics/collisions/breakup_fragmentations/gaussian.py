"""
P(x) = exp(-(x-mu)^2 / 2 sigma^2); mu and sigma are masses
"""


class Gaussian:  # pylint: disable=too-many-instance-attributes
    def __init__(self, mu, sigma, mass_min=0.0, nfmax=None):
        self.particulator = None
        self.mu = mu
        self.sigma = sigma
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
        self.particulator.backend.gauss_fragmentation(
            n_fragment=nf,
            mu=self.mu,
            sigma=self.sigma,
            frag_mass=frag_mass,
            x_plus_y=self.sum_of_masses,
            rand=u01,
            mass_min=self.mass_min,
            nfmax=self.nfmax,
        )
