import numpy as np

from PySDM_examples.utils.basic_simulation import BasicSimulation

from PySDM import Builder, products
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import discretise_multiplicities
from PySDM.initialisation.hygroscopic_equilibrium import equilibrate_wet_radii
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.physics import si


class Simulation(BasicSimulation):
    def __init__(
        self,
        settings,
        product_list=None,
    ):
        self.settings = settings

        environment = Parcel(
            dt=settings.dt,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.p0,
            initial_relative_humidity=settings.RH0,
            T0=settings.T0,
            w=settings.w,
        )

        builder = Builder(
            backend=CPU(settings.formulae),
            n_sd=settings.n_sd,
            environment=environment,
        )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        builder.request_attribute("radius")

        self.builder = builder

        attributes, self.mode_id = self._make_attributes()

        if product_list is None:
            product_list = (
                products.AmbientRelativeHumidity(name="RH"),
                products.Time(name="time"),
                products.AmbientTemperature(name="T"),
            )

        particulator = builder.build(
            attributes=attributes,
            products=product_list,
        )

        super().__init__(
            particulator=particulator,
            output_attributes=["radius"],
        )

    def _make_attributes(self):
        settings = self.settings
        formulae = settings.formulae
        environment = self.builder.particulator.environment

        parcel_volume = environment.mass_of_dry_air / settings.initial_air_density

        n_components = len(settings.aerosol_modes_by_kappa)

        if settings.n_sd % n_components != 0:
            raise ValueError(
                f"settings.n_sd={settings.n_sd} must be divisible by "
                f"n_components={n_components}"
            )

        n_sd_per_component = settings.n_sd // n_components

        dry_volume_parts = []
        kappa_vdry_parts = []
        multiplicity_parts = []
        mode_id_parts = []

        for component_id, (kappa, spectrum) in enumerate(
            settings.aerosol_modes_by_kappa.items()
        ):
            r_dry, concentration = spectral_sampling.Logarithmic(
                spectrum=spectrum
            ).sample_deterministic(n_sd_per_component)

            v_dry = formulae.trivia.volume(radius=r_dry)

            dry_volume_parts.append(v_dry)
            kappa_vdry_parts.append(kappa * v_dry)

            multiplicity_parts.append(
                discretise_multiplicities(concentration * parcel_volume)
            )

            mode_id_parts.append(
                np.full(n_sd_per_component, component_id, dtype=np.int64)
            )

        dry_volume = np.concatenate(dry_volume_parts)
        kappa_times_dry_volume = np.concatenate(kappa_vdry_parts)
        multiplicity = np.concatenate(multiplicity_parts)
        mode_id = np.concatenate(mode_id_parts)

        r_wet = equilibrate_wet_radii(
            r_dry=formulae.trivia.radius(volume=dry_volume),
            environment=environment,
            kappa_times_dry_volume=kappa_times_dry_volume,
        )

        attributes = {
            "multiplicity": multiplicity,
            "dry volume": dry_volume,
            "kappa times dry volume": kappa_times_dry_volume,
            "volume": formulae.trivia.volume(radius=r_wet),
        }

        self._sanity_check_attributes(attributes, mode_id, parcel_volume)

        return attributes, mode_id

    def _sanity_check_attributes(self, attributes, mode_id, volume):
        for attribute in attributes.values():
            assert attribute.shape[0] == self.settings.n_sd

        assert mode_id.shape[0] == self.settings.n_sd

        assert np.all(attributes["multiplicity"] > 0)
        assert np.all(attributes["dry volume"] > 0)
        assert np.all(attributes["volume"] >= attributes["dry volume"])

        kappa_eff = attributes["kappa times dry volume"] / attributes["dry volume"]

        for component_id, kappa in enumerate(
            self.settings.aerosol_modes_by_kappa.keys()
        ):
            mask = mode_id == component_id
            assert np.any(mask)
            assert np.allclose(kappa_eff[mask], kappa)

        np.testing.assert_allclose(
            np.sum(attributes["multiplicity"]) / volume,
            self.settings.total_aerosol_concentration,
            rtol=1e-2,
        )

    def run(self):
        return self._run(
            nt=self.settings.output_interval * self.settings.output_points,
            steps_per_output_interval=self.settings.output_interval,
        )
