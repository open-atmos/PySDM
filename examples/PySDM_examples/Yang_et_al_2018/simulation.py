import PySDM.products as PySDM_products
from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel


class Simulation:
    def __init__(self, settings, backend=CPU):
        dt_output = (
            settings.total_time / settings.n_steps
        )  # TODO #334 overwritten in notebook
        self.n_substeps = 1  # TODO #334 use condensation substeps
        while dt_output / self.n_substeps >= settings.dt_max:
            self.n_substeps += 1
        self.formulae = Formulae(
            condensation_coordinate=settings.coord,
            saturation_vapour_pressure="AugustRocheMagnus",
        )
        self.bins_edges = self.formulae.trivia.volume(settings.r_bins_edges)
        builder = Builder(backend=backend(formulae=self.formulae), n_sd=settings.n_sd)
        builder.set_environment(
            Parcel(
                dt=dt_output / self.n_substeps,
                mass_of_dry_air=settings.mass_of_dry_air,
                p0=settings.p0,
                q0=settings.q0,
                T0=settings.T0,
                w=settings.w,
                z0=settings.z0,
            )
        )

        environment = builder.particulator.environment
        builder.add_dynamic(AmbientThermodynamics())
        condensation = Condensation(
            adaptive=settings.adaptive,
            rtol_x=settings.rtol_x,
            rtol_thd=settings.rtol_thd,
            dt_cond_range=settings.dt_cond_range,
        )
        builder.add_dynamic(condensation)

        products = [
            PySDM_products.ParticleSizeSpectrumPerVolume(
                name="Particles Wet Size Spectrum",
                radius_bins_edges=settings.r_bins_edges,
            ),
            PySDM_products.CondensationTimestepMin(name="dt_cond_min"),
            PySDM_products.CondensationTimestepMax(name="dt_cond_max"),
            PySDM_products.RipeningRate(),
        ]

        attributes = environment.init_attributes(
            n_in_dv=settings.n, kappa=settings.kappa, r_dry=settings.r_dry
        )

        self.particulator = builder.build(attributes, products)

        self.n_steps = settings.n_steps

    def save(self, output):
        _sp = self.particulator
        cell_id = 0
        output["r_bins_values"].append(
            _sp.products["Particles Wet Size Spectrum"].get()
        )
        volume = _sp.attributes["volume"].to_ndarray()
        output["r"].append(self.formulae.trivia.radius(volume=volume))
        output["S"].append(_sp.environment["RH"][cell_id] - 1)
        output["qv"].append(_sp.environment["qv"][cell_id])
        output["T"].append(_sp.environment["T"][cell_id])
        output["z"].append(_sp.environment["z"][cell_id])
        output["t"].append(_sp.environment["t"][cell_id])
        output["dt_cond_max"].append(_sp.products["dt_cond_max"].get()[cell_id].copy())
        output["dt_cond_min"].append(_sp.products["dt_cond_min"].get()[cell_id].copy())
        output["ripening rate"].append(
            _sp.products["ripening rate"].get()[cell_id].copy()
        )

    def run(self):
        output = {
            "r": [],
            "S": [],
            "z": [],
            "t": [],
            "qv": [],
            "T": [],
            "r_bins_values": [],
            "dt_cond_max": [],
            "dt_cond_min": [],
            "ripening rate": [],
        }

        self.save(output)
        for _ in range(self.n_steps):
            self.particulator.run(self.n_substeps)
            self.save(output)
        return output
