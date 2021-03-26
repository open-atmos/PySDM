from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.physics import si
from PySDM.dynamics import AmbientThermodynamics, Condensation, AqueousChemistry
from PySDM.dynamics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS, GASEOUS_COMPOUNDS
import PySDM.products as PySDM_products
import numpy as np


class Simulation:
    def __init__(self, settings):
        env = Parcel(dt=settings.dt, mass_of_dry_air=settings.mass_of_dry_air, p0=settings.p0, q0=settings.q0,
                     T0=settings.T0, w=settings.w, g=settings.g)

        builder = Builder(n_sd=settings.n_sd, backend=CPU)
        builder.set_environment(env)

        attributes = env.init_attributes(
            n_in_dv=settings.n_in_dv,
            kappa=settings.kappa,
            r_dry=settings.r_dry
        )
        attributes = {**attributes, **settings.starting_amounts, 'pH': np.zeros(settings.n_sd)}

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation(kappa=settings.kappa))
        builder.add_dynamic(AqueousChemistry(
            settings.ENVIRONMENT_MOLE_FRACTIONS,
            system_type=settings.system_type,
            n_substep=settings.n_substep
        ))

        products = [
            PySDM_products.RelativeHumidity(),
            PySDM_products.WaterMixingRatio(name='ql', description_prefix='liquid', radius_range=[1*si.um, np.inf]),
            PySDM_products.ParcelDisplacement(),
            PySDM_products.Pressure(),
            PySDM_products.Temperature(),
            PySDM_products.DryAirDensity(),
            PySDM_products.WaterVapourMixingRatio(),
            PySDM_products.Time(),
            *[PySDM_products.AqueousMoleFraction(compound) for compound in AQUEOUS_COMPOUNDS.keys()],
            *[PySDM_products.GaseousMoleFraction(compound) for compound in GASEOUS_COMPOUNDS.keys()],
            PySDM_products.pH(radius_range=settings.cloud_radius_range),
            PySDM_products.TotalDryMassMixingRatio(settings.DRY_RHO),
            PySDM_products.PeakSupersaturation(),
            PySDM_products.CloudDropletConcentration(radius_range=settings.cloud_radius_range)
        ]

        self.core = builder.build(attributes=attributes, products=products)
        self.settings = settings

    def _save(self, output):
        for k, v in self.core.products.items():
            value = v.get()
            if isinstance(value, np.ndarray):
                value = value[0]
            output[k].append(value)

    def run(self):
        output = {k: [] for k in self.core.products.keys()}
        self._save(output)
        for _ in range(0, self.settings.nt+1, self.settings.steps_per_output_interval):
            self.core.run(steps=self.settings.steps_per_output_interval)
            self._save(output)
        return output
