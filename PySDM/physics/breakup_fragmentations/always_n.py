"""
Created at 13.05.2021 by edejong
"""

import numpy as np


class AlwaysN:

    def __init__(self, n):
        self.core = None
        self.N = n
        

    def __call__(self, output, is_first_in_pair):
        output *= 0
        output += self.N
        
        
    def register(self, builder):
        self.core = builder.core