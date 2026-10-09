"""
#TODO #744
"""


class ConstantK:
    def __init__(self):
        self.particulator = None

    def __call__(self, output, is_first_in_pair):
        output.fill(self.particulator.formulae.constants.CONSTANTK_a)

    def register(self, particulator):
        self.particulator = particulator
