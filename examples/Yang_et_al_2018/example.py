"""
Created at 25.11.2019

@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.particles import Particles as Particles
from PySDM.simulation.dynamics.condensation import Condensation
from PySDM.simulation.initialisation import spectral_discretisation
from PySDM.simulation.environment.adiabatic_parcel import AdiabaticParcel
from PySDM.utils import Physics
from examples.Yang_et_al_2018.setup import Setup


class Simulation():
    def __init__(self):
        setup = Setup()
        self.particles = Particles(backend=setup.backend, n_sd=setup.n_sd)
        r, n = spectral_discretisation.logarithmic(setup.n_sd, setup.spectrum_per_mass_of_dry_air, (setup.r_min, setup.r_max))
        x = Physics.r2x(r)
        self.particles.create_state_0d(n=n, extensive={'x': x}, intensive={})

    def run(self):
        pass


if __name__ == '__main__':
    Simulation().run()
