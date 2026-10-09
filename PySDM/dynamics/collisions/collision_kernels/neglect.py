"""
kernel for disabling collision
"""


class Neglect:
    def __init__(self):
        self.particulator = None

    def __call__(self, output, is_first_in_pair):
        output.fill(0)

    def register(self, builder):
        self.particulator = builder.particulator
