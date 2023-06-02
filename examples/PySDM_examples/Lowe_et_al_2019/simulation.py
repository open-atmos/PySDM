import numpy as np
from PySDM_examples.utils import BasicSimulation

import PySDM.products as PySDM_products
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.spectra import Sum


class Simulation(BasicSimulation):
    def __init__(self, settings, products=None):
        env = Parcel(
            dt=settings.dt,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.p0,
            q0=settings.q0,
            T0=settings.T0,
            w=settings.w,
        )
        n_sd = settings.n_sd_per_mode * len(settings.aerosol.modes)
        builder = Builder(n_sd=n_sd, backend=CPU(formulae=settings.formulae))
        builder.set_environment(env)

        attributes = {
            "dry volume": np.empty(0),
            "dry volume organic": np.empty(0),
            "kappa times dry volume": np.empty(0),
            "n": np.ndarray(0),
        }
        for mode in settings.aerosol.modes:
            r_dry, n_in_dv = settings.spectral_sampling(
                spectrum=mode["spectrum"]
            ).sample(settings.n_sd_per_mode)
            V = settings.mass_of_dry_air / settings.rho0
            N = n_in_dv * V
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes["n"] = np.append(attributes["n"], N)
            attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
            attributes["dry volume organic"] = np.append(
                attributes["dry volume organic"], mode["f_org"] * v_dry
            )
            attributes["kappa times dry volume"] = np.append(
                attributes["kappa times dry volume"],
                v_dry * mode["kappa"][settings.model],
            )
        for attribute in attributes.values():
            assert attribute.shape[0] == n_sd

        np.testing.assert_approx_equal(
            np.sum(attributes["n"]) / V,
            Sum(
                tuple(
                    settings.aerosol.modes[i]["spectrum"]
                    for i in range(len(settings.aerosol.modes))
                )
            ).norm_factor,
            significant=5,
        )
        r_wet = equilibrate_wet_radii(
            r_dry=settings.formulae.trivia.radius(volume=attributes["dry volume"]),
            environment=env,
            kappa_times_dry_volume=attributes["kappa times dry volume"],
            f_org=attributes["dry volume organic"] / attributes["dry volume"],
        )
        attributes["volume"] = settings.formulae.trivia.volume(radius=r_wet)

        if settings.model == "Constant":
            del attributes["dry volume organic"]

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        products = products or (
            PySDM_products.ParcelDisplacement(name="z"),
            PySDM_products.Time(name="t"),
            PySDM_products.PeakSupersaturation(unit="%", name="S_max"),
            PySDM_products.AmbientRelativeHumidity(unit="%", name="RH"),
            PySDM_products.ParticleConcentration(
                name="n_c_cm3", unit="cm^-3", radius_range=settings.cloud_radius_range
            ),
            PySDM_products.ParticleSizeSpectrumPerVolume(
                radius_bins_edges=settings.wet_radius_bins_edges
            ),
            PySDM_products.ActivableFraction(),
        )

        particulator = builder.build(attributes=attributes, products=products)
        if settings.scipy_ode_solver:
            scipy_ode_condensation_solver.patch_particulator(particulator)
        self.settings = settings
        super().__init__(particulator=particulator)

    def _save_scalars(self, output):
        for k, v in self.particulator.products.items():
            if len(v.shape) > 1 or k == "activable fraction":
                continue
            value = v.get()
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value[0]
            output[k].append(value)

    def _save_spectrum(self, output):
        value = self.particulator.products["particle size spectrum per volume"].get()
        output["spectrum"] = value

    def run(self):
        output = {k: [] for k in self.particulator.products}
        for step in self.settings.output_steps:
            self.particulator.run(step - self.particulator.n_steps)
            self._save_scalars(output)
        self._save_spectrum(output)
        if self.settings.scipy_ode_solver:
            output["Activated Fraction"] = self.particulator.products[
                "activable fraction"
            ].get(S_max=np.nanmax(output["RH"]) - 100)
        else:
            output["Activated Fraction"] = self.particulator.products[
                "activable fraction"
            ].get(S_max=np.nanmax(output["S_max"]))
        return output
