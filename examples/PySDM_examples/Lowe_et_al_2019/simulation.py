import numpy as np
from PySDM_examples.utils import BasicSimulation

import PySDM.products as PySDM_products
from PySDM import Builder
from PySDM.backends import CPU
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
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            T0=settings.T0,
            w=settings.w,
        )
        n_sd = settings.n_sd_per_mode * len(settings.aerosol.modes)
        builder = Builder(
            n_sd=n_sd,
            backend=CPU(
                formulae=settings.formulae, override_jit_flags={"parallel": False}
            ),
            environment=env,
        )

        attributes = {
            "dry volume": np.empty(0),
            "dry volume organic": np.empty(0),
            "kappa times dry volume": np.empty(0),
            "multiplicity": np.ndarray(0),
        }
        initial_volume = settings.mass_of_dry_air / settings.rho0
        for mode in settings.aerosol.modes:
            r_dry, n_in_dv = settings.spectral_sampling(
                spectrum=mode["spectrum"]
            ).sample(settings.n_sd_per_mode)
            v_dry = settings.formulae.trivia.volume(radius=r_dry)
            attributes["multiplicity"] = np.append(
                attributes["multiplicity"], n_in_dv * initial_volume
            )
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
            np.sum(attributes["multiplicity"]) / initial_volume,
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
            PySDM_products.ActivatedParticleConcentration(
                name="CDNC_cm3",
                unit="cm^-3",
                count_activated=True,
                count_unactivated=False,
            ),
            PySDM_products.ParticleSizeSpectrumPerVolume(
                radius_bins_edges=settings.wet_radius_bins_edges
            ),
            PySDM_products.ActivableFraction(name="Activated Fraction"),
            PySDM_products.WaterMixingRatio(),
            PySDM_products.AmbientDryAirDensity(name="rhod"),
            PySDM_products.ActivatedEffectiveRadius(
                name="reff", count_activated=True, count_unactivated=False
            ),
            PySDM_products.ParcelLiquidWaterPath(
                name="lwp", count_activated=True, count_unactivated=False
            ),
            PySDM_products.CloudOpticalDepth(name="tau"),
            PySDM_products.CloudAlbedo(name="albedo"),
        )

        particulator = builder.build(attributes=attributes, products=products)
        self.settings = settings
        super().__init__(particulator=particulator)

    def _save_scalars(self, output):
        for k, v in self.particulator.products.items():
            if len(v.shape) > 1 or k in ("lwp", "Activated Fraction", "tau", "albedo"):
                continue
            value = v.get()
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value[0]
            output[k].append(value)

    def _save_final_timestep_products(self, output):
        output["spectrum"] = self.particulator.products[
            "particle size spectrum per volume"
        ].get()

        for name, args_call in {
            "Activated Fraction": lambda: {"S_max": np.nanmax(output["S_max"])},
            "lwp": lambda: {},
            "tau": lambda: {
                "effective_radius": output["reff"][-1],
                "liquid_water_path": output["lwp"][0],
            },
            "albedo": lambda: {"optical_depth": output["tau"]},
        }.items():
            output[name] = self.particulator.products[name].get(**args_call())

    def run(self):
        output = {k: [] for k in self.particulator.products}
        for step in self.settings.output_steps:
            self.particulator.run(step - self.particulator.n_steps)
            self._save_scalars(output)
        self._save_final_timestep_products(output)
        return output
