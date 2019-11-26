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
        self.particles = Particles(backend=setup.backend, n_sd=setup.n_sd, dt=setup.dt)
        r, n = spectral_discretisation.logarithmic(setup.n_sd, setup.spectrum, (setup.r_min, setup.r_max))
        x = Physics.r2x(r)
        self.particles.create_state_0d(n=n, extensive={'x': x}, intensive={})
        self.particles.set_environment(AdiabaticParcel, (setup.mass, setup.p0, setup.q0, setup.T0, setup.w))
        self.particles.add_dynamics(Condensation, (setup.kappa, ))

    def run(self):
        self.particles.run(1)


if __name__ == '__main__':
    Simulation().run()
