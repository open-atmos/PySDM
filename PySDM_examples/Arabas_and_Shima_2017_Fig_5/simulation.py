"""
Created at 29.11.2019
"""

import numpy as np

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import AmbientThermodynamics
from PySDM.dynamics import Condensation
from PySDM.environments import Parcel
from PySDM.physics import formulae as phys
from PySDM.initialisation.r_wet_init import r_wet_init
from PySDM.physics import constants as const
from PySDM.products.state import ParticleMeanRadius
from PySDM.products.dynamics.condensation import CondensationTimestep


class Simulation:
    def __init__(self, settings, backend=CPU):
        t_half = settings.z_half / settings.w_avg

        dt_output = (2 * t_half) / settings.n_output
        self.n_substeps = 1
        while dt_output / self.n_substeps >= settings.dt_max:  # TODO dt_max
            self.n_substeps += 1

        builder = Builder(backend=backend, n_sd=1)
        builder.set_environment(Parcel(
            dt=dt_output / self.n_substeps,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.p0,
            q0=settings.q0,
            T0=settings.T0,
            w=settings.w
        ))

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation(
            kappa=settings.kappa,
            rtol_x=settings.rtol_x,
            rtol_thd=settings.rtol_thd,
        ))
        attributes = {}
        r_dry = np.array([settings.r_dry])
        attributes['dry volume'] = phys.volume(radius=r_dry)
        attributes['n'] = np.array([settings.n_in_dv], dtype=np.int64)
        environment = builder.core.environment
        r_wet = r_wet_init(r_dry, environment, np.zeros_like(attributes['n']), settings.kappa)
        attributes['volume'] = phys.volume(radius=r_wet)
        products = [ParticleMeanRadius(), CondensationTimestep()]

        self.core = builder.build(attributes, products)

        self.n_output = settings.n_output

    def save(self, output):
        cell_id = 0
        output["r"].append(self.core.products['radius_m1'].get(unit=const.si.metre)[cell_id])
        output["S"].append(self.core.environment["RH"][cell_id] - 1)
        output["z"].append(self.core.environment["z"][cell_id])
        output["t"].append(self.core.environment["t"][cell_id])
        output["dt"].append(self.core.products['dt_cond'].get()[cell_id])

    def run(self):
        output = {"r": [], "S": [], "z": [], "t": [], "dt": []}

        self.save(output)
        for step in range(self.n_output):
            self.core.run(self.n_substeps)
            self.save(output)

        return output
