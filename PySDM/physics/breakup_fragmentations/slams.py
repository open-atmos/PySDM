"""
Based on 
Created at 18.05.21 by edejong
"""

import numpy as np


class SLAMS:

    def __init__(self):
        self.particulator = None
        self.p_vec = None

    def __call__(self, output, u01, is_first_in_pair):
        self.particulator.backend.slams_fragmentation(output, self.p_vec, u01)
        
    def register(self, builder):
        self.particulator = builder.particulator
        self.p_vec = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)