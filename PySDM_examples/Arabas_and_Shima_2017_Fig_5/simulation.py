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
import PySDM.products as PySDM_products


class Simulation:
    def __init__(self, settings, backend=CPU):
        t_half = settings.z_half / settings.w_avg

        dt_output = (2 * t_half) / settings.n_output
        self.n_substeps = 1
        while dt_output / self.n_substeps >= settings.dt_max:  # TODO #334 dt_max
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
            dt_cond_range=settings.dt_cond_range
        ))
        attributes = {}
        r_dry = np.array([settings.r_dry])
        attributes['dry volume'] = phys.volume(radius=r_dry)
        attributes['n'] = np.array([settings.n_in_dv], dtype=np.int64)
        environment = builder.core.environment
        r_wet = r_wet_init(r_dry, environment, np.zeros_like(attributes['n']), settings.kappa)
        attributes['volume'] = phys.volume(radius=r_wet)
        products = [
            PySDM_products.ParticleMeanRadius(),
            PySDM_products.CondensationTimestepMin(),
            PySDM_products.ParcelDisplacement(),
            PySDM_products.RelativeHumidity(),
            PySDM_products.Time(),
            PySDM_products.ActivatingRate(),
            PySDM_products.DeactivatingRate(),
            PySDM_products.RipeningRate()
        ]

        self.core = builder.build(attributes, products)

        self.n_output = settings.n_output

    def save(self, output):
        cell_id = 0
        output["r"].append(self.core.products['radius_m1'].get(unit=const.si.metre)[cell_id])
        output["dt_cond_min"].append(self.core.products['dt_cond_min'].get()[cell_id])
        output["z"].append(self.core.products["z"].get())
        output["S"].append(self.core.products["RH_env"].get()[cell_id]/100 - 1)
        output["t"].append(self.core.products["t"].get())

        for event in ('activating', 'deactivating', 'ripening'):
            output[event+"_rate"].append(self.core.products[event+'_rate'].get()[cell_id])

    def run(self):
        output = {"r": [], "S": [], "z": [], "t": [], "dt_cond_min": [], "activating_rate": [],
                  "deactivating_rate": [], "ripening_rate": []}

        self.save(output)
        for step in range(self.n_output):
            self.core.run(self.n_substeps)
            self.save(output)

        return output
