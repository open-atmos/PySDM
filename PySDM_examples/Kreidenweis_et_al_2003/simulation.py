from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.physics import constants as const
from PySDM.physics import si
from PySDM.dynamics import AmbientThermodynamics, Condensation, AqueousChemistry
from PySDM.products import (RelativeHumidity, WaterMixingRatio, ParcelDisplacement, Pressure, Temperature,
                            DryAirDensity, WaterVapourMixingRatio, Time, TotalConcentration)
import numpy as np


class Simulation:
    def __init__(self, settings):
        q0 = const.eps * settings.pv0 / (settings.p0 - settings.pv0)
        env = Parcel(dt=settings.dt, mass_of_dry_air=settings.mass_of_dry_air, p0=settings.p0, q0=q0, T0=settings.T0,
                     w=settings.w, g=settings.g)

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
        builder.add_dynamic(AqueousChemistry(settings.ENVIRONMENT_MOLE_FRACTIONS, system_type=settings.system_type))

        products = [
            RelativeHumidity(),
            WaterMixingRatio(name='ql', description_prefix='liquid', radius_range=[1*si.um, np.inf]),
            ParcelDisplacement(),
            Pressure(),
            Temperature(),
            DryAirDensity(),
            WaterVapourMixingRatio(),
            Time(),
            TotalConcentration('SO2')
        ]

        self.core = builder.build(attributes=attributes, products=products)
        self.nt = settings.nt

    def _save(self, output):
        for k, v in self.core.products.items():
            value = v.get()
            if isinstance(value, np.ndarray):
                value = value[0]
            output[k].append(value)

    def run(self, nt=None):
        nt = self.nt if nt is None else nt
        output = {k: [] for k in self.core.products.keys()}
        self._save(output)
        for _ in range(nt):
            self.core.run(steps=1)
            self._save(output)
        return output
