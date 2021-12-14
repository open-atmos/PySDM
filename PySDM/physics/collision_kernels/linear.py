"""
Created at 21.01.2021
"""

class Linear:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.particulator = None

    def __call__(self, output, is_first_in_pair):
        output.sum_pair(self.particulator.particles['volume'], is_first_in_pair)
        output *= self.b
        output += self.a

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute('volume')