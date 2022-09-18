"""
Always produces N fragments in a given collisional breakup
"""

import numpy as np


class AlwaysN:  # pylint: disable=too-many-instance-attributes
    def __init__(self, n, vmin=0.0, nfmax=None):
        self.particulator = None
        self.N = n
        self.N_vec = None
        self.zeros = None
        self.x_plus_y = None
        self.vmax = None
        self.vmin = vmin
        self.nfmax = nfmax

    def __call__(self, nf, frag_size, u01, is_first_in_pair):
        nf *= self.zeros
        nf += self.N_vec
        self.x_plus_y.sum(self.particulator.attributes["volume"], is_first_in_pair)
        self.vmax.max(self.particulator.attributes["volume"], is_first_in_pair)
        frag_size.sum(self.particulator.attributes["volume"], is_first_in_pair)
        frag_size /= self.N

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("volume")
        N_vec_tmp = np.tile([self.N], self.particulator.n_sd // 2)
        zeros_tmp = np.tile([0], self.particulator.n_sd // 2)
        self.N_vec = self.particulator.PairwiseStorage.from_ndarray(N_vec_tmp)
        self.zeros = self.particulator.PairwiseStorage.from_ndarray(zeros_tmp)
        self.x_plus_y = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.vmax = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
