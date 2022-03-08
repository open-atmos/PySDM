"""
Always produces N fragments in a given collisional breakup
"""

import numpy as np


class AlwaysN:
    def __init__(self, n):
        self.particulator = None
        self.N = n
        self.N_vec = None
        self.zeros = None

    def __call__(self, output, u01, is_first_in_pair):
        output *= self.zeros
        output += self.N_vec

    def register(self, builder):
        self.particulator = builder.particulator
        N_vec_tmp = np.tile([self.N], self.particulator.n_sd // 2)
        zeros_tmp = np.tile([0], self.particulator.n_sd // 2)
        self.N_vec = self.particulator.PairwiseStorage.from_ndarray(N_vec_tmp)
        self.zeros = self.particulator.PairwiseStorage.from_ndarray(zeros_tmp)
