"""
Created at 29.11.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.particles import Particles as Particles
from PySDM.simulation.dynamics.condensation import Condensation
from PySDM.simulation.environment.moist_lagrangian_parcel_adiabatic import MoistLagrangianParcelAdiabatic
from PySDM.simulation.physics import formulae as phys
from examples.Arabas_and_Shima_2017_Fig_5.setup import setups
from PySDM.simulation.initialisation.r_wet_init import r_wet_init


class Simulation:
    def __init__(self, setup):
        t_half = setup.z_half / setup.w_avg
        dt = (2 * t_half) / setup.n_steps

        self.particles = Particles(backend=setup.backend, n_sd=1, dt=dt)
        self.particles.set_mesh_0d()
        self.particles.set_environment(MoistLagrangianParcelAdiabatic, {
            "mass_of_dry_air": setup.mass_of_dry_air,
            "p0": setup.p0,
            "q0": setup.q0,
            "T0": setup.T0,
            "w": setup.w
        })

        r_dry = np.array([setup.r_dry])
        x_dry = phys.volume(radius=r_dry)
        n = np.array([setup.n_in_dv], dtype=np.int64)
        r_wet = r_wet_init(r_dry, self.particles.environment, np.zeros_like(n), setup.kappa)
        v_wet = phys.volume(radius=r_wet)
        self.particles.create_state_0d(n=n, extensive={'dry volume': x_dry, 'volume': v_wet}, intensive={})
        self.particles.add_dynamic(Condensation, {"kappa": setup.kappa})

        self.n_steps = setup.n_steps

    # TODO: common with Yang?
    def save(self, output):
        cell_id = 0
        volume = self.particles.state.get_backend_storage('volume')
        volume = self.particles.backend.to_ndarray(volume)
        output["r"].append(phys.radius(volume=volume))
        output["S"].append(self.particles.environment["RH"][cell_id] - 1)
        output["z"].append(self.particles.environment["z"][cell_id])
        output["t"].append(self.particles.environment["t"][cell_id])

    def run(self):
        output = {"r": [], "S": [], "z": [], "t": []}

        self.save(output)
        for step in range(self.n_steps):
            self.particles.run(1)
            self.save(output)

        return output


def main():
    for setup in setups:
        Simulation(setup).run()


if __name__ == '__main__':
    main()
