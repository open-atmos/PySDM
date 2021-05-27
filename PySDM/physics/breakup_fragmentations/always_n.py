"""
Created at 13.05.2021 by edejong
"""

import numpy as np


class AlwaysN:

    def __init__(self, n):
        self.core = None
        self.N = n
        self.N_vec = None
        self.zeros = None
        

    def __call__(self, output, is_first_in_pair):
        output *= self.zeros
        output += self.N_vec
        
    def register(self, builder):
        self.core = builder.core
        N_vec_tmp = np.tile([self.N], self.core.n_sd // 2)
        zeros_tmp = np.tile([0], self.core.n_sd // 2)
        self.N_vec = self.core.PairwiseStorage.from_ndarray(N_vec_tmp)
        self.zeros = self.core.PairwiseStorage.from_ndarray(zeros_tmp)
        