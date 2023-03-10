"""
Always produces fragments of size c in a given collisional breakup
"""


class ConstantSize:  # pylint: disable=too-many-instance-attributes
    def __init__(self, c):
        self.particulator = None
        self.C = c

    def __call__(self, nf, frag_size, u01, is_first_in_pair):
        frag_size[:] = self.C
        nf.sum(self.particulator.attributes["volume"], is_first_in_pair)
        nf /= self.C

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("volume")
