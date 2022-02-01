"""
Created at 30.10.2021
"""
#from PySDM.physics.constants import rho_w, sgm_w
import numpy as np
# TODO: TEST

class Schlottke2010:

    def __init__(self):
        self.particulator = None
        self.pair_tmp = None

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute('volume')
        builder.request_attribute('terminal velocity')
        self.Sc = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.tmp = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.tmp2 = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.We = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)

    def __call__(self, output, is_first_in_pair):
        self.tmp.sum(self.particulator.attributes['volume'], is_first_in_pair)
        self.tmp /= (np.pi / 6)
        
        self.tmp2.distance(self.particulator.attributes['terminal velocity'], is_first_in_pair)
        self.tmp2 **= 2
        self.We.multiply(self.particulator.attributes['volume'], is_first_in_pair)
        self.We /= self.tmp
        self.We *= self.tmp2
        self.We *= (np.pi / 12 * rho_w)
        
        self.Sc[:] = self.tmp[:]
        self.Sc **= (2/3)
        self.Sc *= (np.pi * sgm)

        self.We /= self.Sc
        self.We *= -1.15

        output[:] = np.exp(self.We)