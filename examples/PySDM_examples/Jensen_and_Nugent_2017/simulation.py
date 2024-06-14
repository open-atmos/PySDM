import numpy as np
from PySDM_examples.utils import BasicSimulation
from PySDM_examples.Jensen_and_Nugent_2017.settings import Settings
from PySDM_examples.Jensen_and_Nugent_2017 import table_3
from PySDM import Builder
from PySDM.physics import si
from PySDM.backends import CPU
from PySDM.products import (
    PeakSupersaturation,
    ParcelDisplacement,
    Time,
    ActivatedMeanRadius,
    RadiusStandardDeviation,
)
from PySDM.environments import Parcel
from PySDM.dynamics import Condensation, AmbientThermodynamics, Coalescence
from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.initialisation.sampling.spectral_sampling import Logarithmic

# note: 100 in caption of Table 1
N_SD_NON_GCCN = 100


class Simulation(BasicSimulation):
    def __init__(
        self,
        settings: Settings,
        gccn: bool = False,
        gravitational_coalsecence: bool = False,
    ):
        const = settings.formulae.constants
        pvs_Celsius = settings.formulae.saturation_vapour_pressure.pvs_Celsius
        initial_water_vapour_mixing_ratio = const.eps / (
            settings.p0 / settings.RH0 / pvs_Celsius(settings.T0 - const.T0) - 1
        )

        env = Parcel(
            dt=settings.dt,
            mass_of_dry_air=666 * si.kg,
            p0=settings.p0,
            initial_water_vapour_mixing_ratio=initial_water_vapour_mixing_ratio,
            T0=settings.T0,
            w=settings.vertical_velocity,
            z0=settings.z0,
        )

        n_gccn = np.count_nonzero(table_3.NA) if gccn else 0

        builder = Builder(
            n_sd=N_SD_NON_GCCN + n_gccn,
            backend=CPU(
                formulae=settings.formulae, override_jit_flags={"parallel": False}
            ),
            environment=env,
        )

        additional_derived_attributes = ("radius", "equilibrium supersaturation")
        for additional_derived_attribute in additional_derived_attributes:
            builder.request_attribute(additional_derived_attribute)

        builder.add_dynamic(
            AmbientThermodynamics()
        )  # TODO #1266: order matters here, but error message is not saying it!
        builder.add_dynamic(Condensation())
        if gravitational_coalsecence:
            builder.add_dynamic(Coalescence(collision_kernel=Geometric()))

        self.r_dry, n_in_unit_volume = Logarithmic(
            spectrum=settings.dry_radii_spectrum,
        ).sample(builder.particulator.n_sd - n_gccn)

        if gccn:
            nonzero_concentration_mask = np.nonzero(table_3.NA)
            self.r_dry = np.concatenate(
                [self.r_dry, table_3.RD[nonzero_concentration_mask]]
            )
            n_in_unit_volume = np.concatenate(
                [n_in_unit_volume, table_3.NA[nonzero_concentration_mask]]
            )  # TODO #1266: check which temp, pres, RH assumed in the paper for NA???

        pd0 = settings.formulae.trivia.p_d(
            settings.p0, initial_water_vapour_mixing_ratio
        )
        rhod0 = settings.formulae.state_variable_triplet.rhod_of_pd_T(pd0, settings.T0)

        attributes = env.init_attributes(
            n_in_dv=n_in_unit_volume * env.mass_of_dry_air / rhod0,
            kappa=settings.kappa,
            r_dry=self.r_dry,
        )

        super().__init__(
            builder.build(
                attributes=attributes,
                products=(
                    PeakSupersaturation(name="S_max"),
                    ParcelDisplacement(name="z"),
                    Time(name="t"),
                    ActivatedMeanRadius(
                        name="r_mean_act", count_activated=True, count_unactivated=False
                    ),
                    RadiusStandardDeviation(
                        name="r_std_act", count_activated=True, count_unactivated=False
                    ),
                ),
            )
        )

        # TODO #1266: copied from G & P 2023
        self.output_attributes = {
            attr: tuple([] for _ in range(self.particulator.n_sd))
            for attr in additional_derived_attributes
        }

    def run(
        self, *, n_steps: int = 2250, steps_per_output_interval: int = 10
    ):  # TODO #1266: essentially copied from G & P 2023
        output_products = super()._run(
            nt=n_steps, steps_per_output_interval=steps_per_output_interval
        )
        return {"products": output_products, "attributes": self.output_attributes}

    def _save(self, output):  # TODO #1266: copied from G&P 2023
        for key, attr in self.output_attributes.items():
            attr_data = self.particulator.attributes[key].to_ndarray()
            for drop_id in range(self.particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
        super()._save(output)
