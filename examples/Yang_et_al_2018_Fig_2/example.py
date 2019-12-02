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
from examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM.simulation.initialisation.r_wet_init import r_wet_init

# TODO: the q1 logic from PyCloudParcel?


class Simulation:
    def __init__(self, setup):
        self.r_dry, self.n = spectral_discretisation.logarithmic(setup.n_sd, setup.spectrum, (setup.r_min, setup.r_max))

        # self.r_dry = self.r_dry[50:52]
        # self.n = self.n[50:52]
        # setup.n_sd = 2

        self.particles = Particles(backend=setup.backend, n_sd=setup.n_sd, dt=setup.dt)
        self.particles.set_mesh_0d()
        self.particles.set_environment(AdiabaticParcel, (setup.mass_of_dry_air, setup.p0, setup.q0, setup.T0, setup.w, setup.z0))


        x_dry = Physics.r2x(self.r_dry)
        r_wet = r_wet_init(self.r_dry, self.particles.environment, np.zeros_like(self.n), setup.kappa)
        x_wet = Physics.r2x(r_wet)
        self.particles.create_state_0d(n=self.n, extensive={'dry volume': x_dry, 'x': x_wet}, intensive={})
        self.particles.add_dynamic(Condensation, (setup.kappa,))

        self.n_steps = setup.n_steps

    # TODO: make it common with Arabas_and_Shima_2017
    def run(self):
        output = {"r": [], "S": [], "z": [], "t": [], "qv": [], "T": []}

        # TODO: save t=0
        for step in range(self.n_steps):
            self.particles.run(1)

            # TODO
            cell_id = 0
            x = self.particles.state.get_backend_storage('x')
            x = self.particles.backend.to_ndarray(x)
            output["r"].append(Physics.x2r(x))
            output["S"].append(self.particles.environment["RH"][cell_id]-1)
            output["qv"].append(self.particles.environment["qv"][cell_id])
            output["T"].append(self.particles.environment["T"][cell_id])
            output["z"].append(self.particles.environment["z"][cell_id])
            output["t"].append(self.particles.environment["t"][cell_id])

        return output


if __name__ == '__main__':
    Simulation(setup=Setup()).run()
