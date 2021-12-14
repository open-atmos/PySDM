"""
Created at 21.01.2021
"""

class Linear:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.core = None

    def __call__(self, output, is_first_in_pair):
        output.sum_pair(self.core.particles['volume'], is_first_in_pair)
        output *= self.b
        output += self.a

    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('volume')