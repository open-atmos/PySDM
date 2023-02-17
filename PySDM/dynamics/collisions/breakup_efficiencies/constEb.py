"""
Specifies constant breakup efficiency.
"""


class ConstEb:
    def __init__(self, Eb=1.0):
        self.Eb = Eb
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self, output, is_first_in_pair):
        output.fill(self.Eb)
