"""
P(m; x, y) = nu^2 * (x+y) exp(-m * nu)
nu = 1/m* where m* is a scaling factor for fragment size dist.
see [Feingold et al. 1999](https://doi.org/10.1175/1520-0469(1999)056<4100:TIOGCC>2.0.CO;2)
"""


class Feingold1988:  # pylint: disable=too-many-instance-attributes
    def __init__(self, scale, fragtol=1e-3, mass_min=0.0, nfmax=None):
        self.particulator = None
        self.scale = scale
        self.fragtol = fragtol
        self.mass_min = mass_min
        self.nfmax = nfmax
        self.sum_of_masses = None

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("water mass")
        self.sum_of_masses = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )

    def __call__(self, nf, frag_mass, u01, is_first_in_pair):
        self.sum_of_masses.sum(
            self.particulator.attributes["water mass"], is_first_in_pair
        )
        self.particulator.backend.feingold1988_fragmentation(
            n_fragment=nf,
            scale=self.scale,
            frag_mass=frag_mass,
            x_plus_y=self.sum_of_masses,
            rand=u01,
            fragtol=self.fragtol,
            mass_min=self.mass_min,
            nfmax=self.nfmax,
        )
