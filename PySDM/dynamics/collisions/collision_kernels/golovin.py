"""
Safranov-Golovin kernel with analytic solution
(see [Safranov 1962](http://mi.mathnet.ru/dan27172),
[Golovin 1963](http://mi.mathnet.ru/dan27630))
"""

import numpy as np
from scipy import special


class Golovin:
    def __init__(self, b):
        self.b = b
        self.particulator = None

    def __call__(self, output, is_first_in_pair):
        output.sum(self.particulator.attributes["volume"], is_first_in_pair)
        output *= self.b

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("volume")
