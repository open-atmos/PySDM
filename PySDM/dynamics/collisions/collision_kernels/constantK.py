"""
#TODO #744
"""


class ConstantK:
    def __init__(self, a):
        self.a = a
        self.particulator = None

    def __call__(self, output, is_first_in_pair):
        output.fill(self.a)

    def register(self, builder):
        self.particulator = builder.particulator
