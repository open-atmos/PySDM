"""
Low & List fragmentation probability
Created at 18.05.21 by edejong
"""

import numpy as np


class Straub2010_r3:

    def __init__(self):
        self.core = None
        self.p_vec = None

    def __call__(self, output, u01, is_first_in_pair):
        self.core.backend.straub2010_r3_fragmentation(output, self.p_vec, u01)
        
    def register(self, builder):
        self.core = builder.core
        self.p_vec = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)