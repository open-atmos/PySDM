"""
Created at 29.11.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.particles_builder import ParticlesBuilder
from PySDM.dynamics import Condensation
from PySDM.environments import MoistLagrangianParcelAdiabatic
from PySDM.physics import formulae as phys
from PySDM.initialisation.r_wet_init import r_wet_init
from PySDM.physics import constants as const
from PySDM.state.products.particle_mean_radius import ParticleMeanRadius


class Simulation:
    def __init__(self, setup):
        t_half = setup.z_half / setup.w_avg

        dt_output = (2 * t_half) / setup.n_output
        self.n_substeps = 1
        while dt_output / self.n_substeps >= setup.dt_max:  # TODO dt_max
            self.n_substeps += 1

        particles_builder = ParticlesBuilder(backend=setup.backend, n_sd=1)
        particles_builder.set_environment(MoistLagrangianParcelAdiabatic, {
            "dt": dt_output / self.n_substeps,
            "mass_of_dry_air": setup.mass_of_dry_air,
            "p0": setup.p0,
            "q0": setup.q0,
            "T0": setup.T0,
            "w": setup.w
        })

        particles_builder.register_dynamic(Condensation, {
            "kappa": setup.kappa,
            "rtol_x": setup.rtol_x,
            "rtol_thd": setup.rtol_thd,
        })
        attributes = {}
        r_dry = np.array([setup.r_dry])
        attributes['dry volume'] = phys.volume(radius=r_dry)
        attributes['n'] = np.array([setup.n_in_dv], dtype=np.int64)
        r_wet = r_wet_init(r_dry, particles_builder.particles.environment, np.zeros_like(attributes['n']), setup.kappa)
        attributes['volume'] = phys.volume(radius=r_wet)
        products = {
            ParticleMeanRadius: {}
        }

        self.particles = particles_builder.get_particles(attributes, products)

        self.n_output = setup.n_output

    def save(self, output):
        cell_id = 0
        output["r"].append(self.particles.products['radius_m1'].get(unit=const.si.metre)[cell_id])
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
