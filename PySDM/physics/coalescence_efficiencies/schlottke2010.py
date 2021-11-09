"""
Created at 30.10.2021
"""
from PySDM.physics.constants import rho_w, sgm
import numpy as np
# TODO: TEST

class Schlottke2010:

    def __init__(self):
        self.core = None
        self.pair_tmp = None

    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('volume')
        builder.request_attribute('terminal velocity')
        self.sigma = sgm #constant from LL82a
        self.Sc = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.tmp = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.tmp2 = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.We = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)

    def __call__(self, output, is_first_in_pair):
        self.tmp.sum(self.core.particles['volume'], is_first_in_pair)
        self.tmp /= (np.pi / 6)
        
        self.tmp2.difference(self.core.particles['terminal velocity'])
        self.tmp2 **= 2
        self.We.multiply(self.core.particles['volume'], is_first_in_pair)
        self.We /= self.tmp
        self.We *= self.tmp2
        self.We *= (np.pi / 12 * rho_w)
        
        self.Sc[:] = self.tmp[:]
        self.Sc **= (2/3)
        self.Sc *= (np.pi * self.sigma)

        self.We /= self.Sc

        output[:] = np.exp(-1.15 * self.We)
        print(output.data)