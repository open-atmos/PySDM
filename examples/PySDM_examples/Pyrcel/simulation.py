import numpy as np
from PySDM_examples.utils import BasicSimulation

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si


class Simulation(BasicSimulation):
    def __init__(
        self, settings, products=None, scipy_solver=False, rtol_thd=1e-10, rtol_x=1e-10
    ):
        env = Parcel(
            dt=settings.timestep,
            p0=settings.initial_pressure,
            initial_water_vapour_mixing_ratio=settings.initial_vapour_mixing_ratio,
            T0=settings.initial_temperature,
            w=settings.vertical_velocity,
            mass_of_dry_air=44 * si.kg,
        )
        n_sd = sum(settings.n_sd_per_mode)
        builder = Builder(
            n_sd=n_sd,
            backend=CPU(
                formulae=settings.formulae, override_jit_flags={"parallel": False}
            ),
            environment=env,
        )
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation(rtol_thd=rtol_thd, rtol_x=rtol_x))

        volume = env.mass_of_dry_air / settings.initial_air_density
        attributes = {
            k: np.empty(0)
            for k in ("dry volume", "kappa times dry volume", "multiplicity")
        }
        for i, (kappa, spectrum) in enumerate(settings.aerosol_modes_by_kappa.items()):
            sampling = ConstantMultiplicity(spectrum)
            r_dry, n_per_volume = sampling.sample(settings.n_sd_per_mode[i])
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes["multiplicity"] = np.append(
                attributes["multiplicity"], n_per_volume * volume
            )
            attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
            attributes["kappa times dry volume"] = np.append(
                attributes["kappa times dry volume"], v_dry * kappa
            )
        r_wet = equilibrate_wet_radii(
            r_dry=settings.formulae.trivia.radius(volume=attributes["dry volume"]),
            environment=env,
            kappa_times_dry_volume=attributes["kappa times dry volume"],
        )
        attributes["volume"] = settings.formulae.trivia.volume(radius=r_wet)

        super().__init__(
            particulator=builder.build(attributes=attributes, products=products)
        )
        if scipy_solver:
            scipy_ode_condensation_solver.patch_particulator(self.particulator)

        self.output_attributes = {
            "volume": tuple([] for _ in range(self.particulator.n_sd))
        }
        self.settings = settings

        self.__sanity_checks(attributes, volume)

    def __sanity_checks(self, attributes, volume):
        for attribute in attributes.values():
            assert attribute.shape[0] == self.particulator.n_sd
        np.testing.assert_approx_equal(
            sum(attributes["multiplicity"]) / volume,
            sum(
                mode.norm_factor
                for mode in self.settings.aerosol_modes_by_kappa.values()
            ),
            significant=4,
        )

    def _save(self, output):
        for key, attr in self.output_attributes.items():
            attr_data = self.particulator.attributes[key].to_ndarray()
            for drop_id in range(self.particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
        super()._save(output)

    def run(self):
        output_products = super()._run(
            self.settings.nt, self.settings.steps_per_output_interval
        )
        return {"products": output_products, "attributes": self.output_attributes}
