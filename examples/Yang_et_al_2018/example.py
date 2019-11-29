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
from PySDM.simulation.initialisation.r_wet_init import r_wet_init


class Simulation:
    def __init__(self):
        setup = Setup()
        self.particles = Particles(backend=setup.backend, n_sd=setup.n_sd, dt=setup.dt)
        self.particles.set_mesh_0d()
        self.particles.set_environment(AdiabaticParcel, (setup.mass, setup.p0, setup.q0, setup.T0, setup.w))
        r_dry, n = spectral_discretisation.logarithmic(setup.n_sd, setup.spectrum, (setup.r_min, setup.r_max))
        x_dry = Physics.r2x(r_dry)
        r_wet = r_wet_init(r_dry, self.particles.environment, np.zeros_like(n), setup.kappa)
        x_wet = Physics.r2x(r_wet)
        self.particles.create_state_0d(n=n, extensive={'dry volume': x_dry, 'x': x_wet}, intensive={})
        self.particles.add_dynamic(Condensation, (setup.kappa,))

    def run(self):
        self.particles.run(1)


if __name__ == '__main__':
    Simulation().run()
