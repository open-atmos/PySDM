from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.physics import constants as const
from PySDM.physics import si
from PySDM.dynamics import AmbientThermodynamics, Condensation, AqueousChemistry
from PySDM.products import RelativeHumidity
import numpy as np

# Chemical Conditions
# SO2 at t = 0 	200 (ppt‐v)
# NH3(g) at t = 0 	100 (ppt‐v)
# H2O2 at t = 0 	500 (ppt‐v)


class Simulation:
    def __init__(self, settings):
        q0 = const.eps * settings.pv0 / (settings.p0 - settings.pv0)
        env = Parcel(dt=settings.dt, mass_of_dry_air=settings.mass_of_dry_air, p0=settings.p0, q0=q0, T0=settings.T0, w=.5*si.m/si.s, z0=600*si.m)

        builder = Builder(n_sd=settings.n_sd, backend=CPU)
        builder.set_environment(env)

        attributes = env.init_attributes(
            n_in_dv=settings.n_in_dv,
            kappa=settings.kappa,
            r_dry=settings.r_dry
        )
        attributes = {**attributes, **settings.starting_amounts}

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation(kappa=settings.kappa))
        builder.add_dynamic(AqueousChemistry(settings.ENVIRONMENT_AMOUNTS))
        self.core = builder.build(attributes=attributes, products=[RelativeHumidity()])

    def run(self, steps):
        self.core.run(steps=steps)
