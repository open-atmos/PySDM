""" constant value """


class ConstEc:
    def __init__(self, Ec=1.0):
        self.Ec = Ec
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self, output, is_first_in_pair):
        output.fill(self.Ec)
