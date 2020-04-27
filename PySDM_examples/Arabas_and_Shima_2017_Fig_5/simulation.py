"""
Created at 29.11.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.particles_builder import ParticlesBuilder
from PySDM.simulation.dynamics.condensation.condensation import Condensation
from PySDM.simulation.environment.moist_lagrangian_parcel_adiabatic import MoistLagrangianParcelAdiabatic
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.initialisation.r_wet_init import r_wet_init
from PySDM.simulation.physics import constants as const


class Simulation:
    def __init__(self, setup):
        t_half = setup.z_half / setup.w_avg

        dt_output = (2 * t_half) / setup.n_output
        self.n_substeps = 1
        while dt_output / self.n_substeps >= setup.dt_max:  # TODO dt_max
            self.n_substeps += 1

        particles_builder = ParticlesBuilder(backend=setup.backend, n_sd=1, dt=dt_output / self.n_substeps)
        particles_builder.set_mesh_0d()
        particles_builder.set_environment(MoistLagrangianParcelAdiabatic, {
            "mass_of_dry_air": setup.mass_of_dry_air,
            "p0": setup.p0,
            "q0": setup.q0,
            "T0": setup.T0,
            "w": setup.w
        })

        r_dry = np.array([setup.r_dry])
        x_dry = phys.volume(radius=r_dry)
        n = np.array([setup.n_in_dv], dtype=np.int64)
        r_wet = r_wet_init(r_dry, particles_builder.particles.environment, np.zeros_like(n), setup.kappa)
        v_wet = phys.volume(radius=r_wet)
        particles_builder.create_state_0d(n=n, extensive={'dry volume': x_dry, 'volume': v_wet}, intensive={})
        particles_builder.register_dynamic(Condensation, {
            "kappa": setup.kappa,
            "rtol_x": setup.rtol_x,
            "rtol_thd": setup.rtol_thd,
        })
        self.particles = particles_builder.get_particles()

        self.n_output = setup.n_output

    def save(self, output):
        cell_id = 0
        output["r"].append(self.particles.products['radius_m1'].get(unit=const.si.metre))
        output["S"].append(self.particles.environment["RH"][cell_id] - 1)
        output["z"].append(self.particles.environment["z"][cell_id])
        output["t"].append(self.particles.environment["t"][cell_id])

    def run(self):
        output = {"r": [], "S": [], "z": [], "t": []}

        self.save(output)
        for step in range(self.n_output):
            self.particles.run(self.n_substeps)
            self.save(output)

        return output
