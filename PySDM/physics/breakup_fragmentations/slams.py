"""
Created at 18.05.21 by edejong
"""

import numpy as np


class SLAMS:

    def __init__(self):
        self.core = None
        self.N_vec = None
        self.zeros = None

    def __call__(self, output, is_first_in_pair):
        p = 0.0
        r = np.random.rand()       # TODO: optimize random generator outside the fragmentation function
        nf = 1
        for i in range(22):
            p += 0.91 * (i + 2)**(-1.56)
            if (r < p):
                nf = i + 2
                break
        output *= self.zeros
        output += self.N_vec
        output *= nf
        print(nf, flush=True)

    def register(self, builder):
        self.core = builder.core
        N_vec_tmp = np.tile([1], self.core.n_sd // 2)
        zeros_tmp = np.tile([0], self.core.n_sd // 2)
        self.N_vec = self.core.PairwiseStorage.from_ndarray(N_vec_tmp)
        self.zeros = self.core.PairwiseStorage.from_ndarray(zeros_tmp)