"""
Created at 2019
"""

from PySDM.mesh import Mesh
from PySDM.builder import Builder
from PySDM.initialisation.spectra import Spectrum


class Box:

    def __init__(self, dt, dv=None):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)
        self.core = None

    def register(self, builder: Builder):
        self.core = builder.core

    def init_attributes(self, *, initial_spectrum: Spectrum, sampling_scheme: callable, sampling_range: tuple):
        attributes = {}
        attributes['volume'], attributes['n'] = sampling_scheme(
             n_sd=self.core.n_sd, spectrum=initial_spectrum, range=sampling_range)
        return attributes
