"""
Always produces fragments of mass c in a given collisional breakup
"""


class ConstantMass:  # pylint: disable=too-many-instance-attributes
    def __init__(self, c):
        self.particulator = None
        self.C = c

    def __call__(self, nf, frag_mass, u01, is_first_in_pair):
        frag_mass[:] = self.C
        nf.sum(self.particulator.attributes["water mass"], is_first_in_pair)
        nf /= self.C

    def register(self, builder):
        self.particulator = builder.particulator
