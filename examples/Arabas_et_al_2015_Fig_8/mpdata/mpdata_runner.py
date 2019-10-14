"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from examples.Arabas_et_al_2015_Fig_8.mpdata.mpdata import MPDATA
from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import ScalarField, VectorField
import numba
from numba import prange


class MPDATARunner:
    def __init__(self, init_values: dict, courant_field: tuple, n_iters: int):
        # TODO check values shape
        self.halo = 1

        self.C_physical = VectorField(*courant_field, halo=self.halo)

        self.mpdata = {}
        self.keys = {}
        for i, key in enumerate(init_values):
            state = ScalarField(init_values[key], halo=self.halo)
            self.mpdata[key] = MPDATA(state, self.C_physical, n_iters, self.halo)
            self.keys[i] = key

    # @numba.jit(parallel=True)
    def step(self):
        for i in range(len(self.keys)):
            key = self.keys[i]
            self.mpdata[key].step()

    def psi(self, key):  # TODO: way to access previous value
        shape = self.mpdata[key].state.data.shape
        i = slice(self.halo, shape[0] - self.halo)
        j = slice(self.halo, shape[1] - self.halo)
        return self.mpdata[key].state.data[(i, j)]