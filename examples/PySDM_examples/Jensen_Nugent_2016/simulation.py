from PySDM_examples.utils import BasicSimulation
from PySDM_examples.Jensen_Nugent_2016 import Settings
from PySDM import Builder, Formulae
from PySDM.physics import si
from PySDM.backends import CPU
from PySDM.products import PeakSupersaturation, ParcelDisplacement
from PySDM.environments import Parcel
from PySDM.dynamics import Condensation, AmbientThermodynamics
from PySDM.initialisation.sampling.spectral_sampling import Logarithmic


class Simulation(BasicSimulation):
    def __init__(self, settings: Settings):
        const = settings.formulae.constants
        pvs_Celsius = settings.formulae.saturation_vapour_pressure.pvs_Celsius
        initial_water_vapour_mixing_ratio = const.eps / (
            settings.p0 / settings.RH0 / pvs_Celsius(settings.T0 - const.T0) - 1
        )

        env = Parcel(
            dt=1 * si.s,  # TODO: not found in the paper yet
            mass_of_dry_air=666 * si.kg,
            p0=settings.p0,
            initial_water_vapour_mixing_ratio=initial_water_vapour_mixing_ratio,
            T0=settings.T0,
            w=settings.vertical_velocity,
            z0=600 * si.m,
        )

        builder = Builder(
            n_sd=100, backend=CPU(formulae=settings.formulae), environment=env
        )
        builder.request_attribute("radius")

        builder.add_dynamic(
            AmbientThermodynamics()
        )  # TODO: order matters here, but error message is not saying it!
        builder.add_dynamic(Condensation())

        self.r_dry, n_in_unit_volume = Logarithmic(
            spectrum=settings.dry_radii_spectrum, size_range=(0.01 * si.um, 0.5 * si.um)
        ).sample(builder.particulator.n_sd)

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
                ),
            )
        )

        # TODO: copied from G & P 2023
        self.output_attributes = {
            "radius": tuple([] for _ in range(self.particulator.n_sd)),
        }

    def run(
        self, *, n_steps, steps_per_output_interval
    ):  # TODO: essentially copied from G & P 2023
        output_products = super()._run(n_steps, steps_per_output_interval)
        return {"products": output_products, "attributes": self.output_attributes}

    def _save(self, output):  # TODO: copied from G&P 2023
        for key, attr in self.output_attributes.items():
            attr_data = self.particulator.attributes[key].to_ndarray()
            for drop_id in range(self.particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
        super()._save(output)
