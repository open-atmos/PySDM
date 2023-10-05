"""
Always produces N fragments in a given collisional breakup
"""


class AlwaysN:  # pylint: disable=too-many-instance-attributes
    def __init__(self, n, mass_min=0.0, nfmax=None):
        self.particulator = None
        self.N = n
        self.mass_min = mass_min
        self.nfmax = nfmax

    def __call__(self, nf, frag_mass, u01, is_first_in_pair):
        nf.fill(self.N)
        frag_mass.sum(self.particulator.attributes["water mass"], is_first_in_pair)
        frag_mass /= self.N

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("water mass")
