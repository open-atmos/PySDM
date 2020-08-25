"""
Created at 24.08.2020 by edejong
"""

from ._gravitational import Gravitational
from PySDM.physics import constants as const

class Constant:
    def __init__(self, kernel_const):
        self.kernel_const = kernel_const
        
    def __call__(self, output, is_first_in_pair):
        # TODO: stop stupidly summing over all particles
        output.sum_pair(self.core.state['volume'],is_first_in_pair)
        output *= 0
        output += self.kernel_const
        
    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('volume')
        
# TODO: linear/polynomial kernel