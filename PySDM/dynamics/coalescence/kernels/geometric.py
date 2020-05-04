"""
Created at 04.05.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Geometric:
    def __init__(self, b, x='volume'):
        self.b = b
        self.x = x
        self.particles = None

    def __call__(self, output, is_first_in_pair):
        self.particles.subtract_pair(output, self.x, is_first_in_pair)
        self.particles.abs(output)
        self.particles.backend.multiply(output, self.b)
