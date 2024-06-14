import numpy as np
from PySDM_examples.utils import BasicSimulation

import PySDM.products as PySDM_products
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, AqueousChemistry, Condensation
from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS, GASEOUS_COMPOUNDS
from PySDM.environments import Parcel
from PySDM.physics import si


class Simulation(BasicSimulation):
    def __init__(self, settings, products=None):
        env = Parcel(
            dt=settings.dt,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.p0,
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            T0=settings.T0,
            w=settings.w,
        )

        builder = Builder(
            n_sd=settings.n_sd,
            backend=CPU(
                formulae=settings.formulae, override_jit_flags={"parallel": False}
            ),
            environment=env,
        )

        attributes = env.init_attributes(
            n_in_dv=settings.n_in_dv,
            kappa=settings.kappa,
            r_dry=settings.r_dry,
            include_dry_volume_in_attribute=False,
        )
        attributes = {
            **attributes,
            **settings.starting_amounts,
        }

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(
            AqueousChemistry(
                environment_mole_fractions=settings.ENVIRONMENT_MOLE_FRACTIONS,
                system_type=settings.system_type,
                n_substep=settings.n_substep,
                dry_rho=settings.DRY_RHO,
                dry_molar_mass=settings.dry_molar_mass,
            )
        )

        products = products or (
            PySDM_products.AmbientRelativeHumidity(name="RH", unit="%"),
            PySDM_products.WaterMixingRatio(
                name="liquid water mixing ratio",
                radius_range=[1 * si.um, np.inf],
                unit="g/kg",
            ),
            PySDM_products.ParcelDisplacement(name="z"),
            PySDM_products.AmbientPressure(name="p"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.AmbientDryAirDensity(name="rhod"),
            PySDM_products.AmbientWaterVapourMixingRatio(
                name="water vapour mixing ratio",
                var="water_vapour_mixing_ratio",
                unit="g/kg",
            ),
            PySDM_products.Time(name="t"),
            *(
                PySDM_products.AqueousMoleFraction(
                    comp, unit="ppb", name=f"aq_{comp}_ppb"
                )
                for comp in AQUEOUS_COMPOUNDS
            ),
            *(
                PySDM_products.GaseousMoleFraction(
                    comp, unit="ppb", name=f"gas_{comp}_ppb"
                )
                for comp in GASEOUS_COMPOUNDS
            ),
            PySDM_products.Acidity(
                name="pH_pH_number_weighted",
                radius_range=settings.cloud_radius_range,
                weighting="number",
                attr="pH",
            ),
            PySDM_products.Acidity(
                name="pH_pH_volume_weighted",
                radius_range=settings.cloud_radius_range,
                weighting="volume",
                attr="pH",
            ),
            PySDM_products.Acidity(
                name="pH_conc_H_number_weighted",
                radius_range=settings.cloud_radius_range,
                weighting="number",
                attr="conc_H",
            ),
            PySDM_products.Acidity(
                name="pH_conc_H_volume_weighted",
                radius_range=settings.cloud_radius_range,
                weighting="volume",
                attr="conc_H",
            ),
            PySDM_products.TotalDryMassMixingRatio(
                settings.DRY_RHO, name="q_dry", unit="ug/kg"
            ),
            PySDM_products.PeakSupersaturation(unit="%", name="S_max"),
            PySDM_products.ParticleSpecificConcentration(
                radius_range=settings.cloud_radius_range, name="n_c_mg", unit="mg^-1"
            ),
            PySDM_products.AqueousMassSpectrum(
                key="S_VI",
                dry_radius_bins_edges=settings.dry_radius_bins_edges,
                name="dm_S_VI/dlog_10(dry diameter)",
                unit='ug / m^3"',
            ),
        )

        particulator = builder.build(attributes=attributes, products=products)
        self.settings = settings
        super().__init__(particulator=particulator)

    def run(self):
        return super()._run(self.settings.nt, self.settings.steps_per_output_interval)
