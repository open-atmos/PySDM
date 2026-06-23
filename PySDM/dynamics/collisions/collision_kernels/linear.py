"""
TODO #744
"""


class Linear:
    def __init__(self):
        self.particulator = None

    def __call__(self, output, is_first_in_pair):
        output.sum_pair(self.particulator.attributes["volume"], is_first_in_pair)
        output *= self.particulator.formulae.constants.LINEAR_b
        output += self.particulator.formulae.constants.LINEAR_a

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("volume")
