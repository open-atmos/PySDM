"""
Created at 23.04.2020
"""


from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import AmbientThermodynamics
from PySDM.dynamics import Condensation
from PySDM.environments import Parcel
from PySDM.physics import formulae as phys
from PySDM.products.state import ParticlesWetSizeSpectrum
from PySDM.products.dynamics.condensation import CondensationTimestep
from PySDM.products.dynamics.condensation.ripening_rate import RipeningRate


class Simulation:

    def __init__(self, settings, backend=CPU):

        dt_output = settings.total_time / settings.n_steps  # TODO: overwritten in jupyter example
        self.n_substeps = 1  # TODO
        while (dt_output / self.n_substeps >= settings.dt_max):
            self.n_substeps += 1
        self.bins_edges = phys.volume(settings.r_bins_edges)
        builder = Builder(backend=backend, n_sd=settings.n_sd)
        builder.set_environment(Parcel(
            dt=dt_output / self.n_substeps,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.p0,
            q0=settings.q0,
            T0=settings.T0,
            w=settings.w,
            z0=settings.z0
        ))

        environment = builder.core.environment
        builder.add_dynamic(AmbientThermodynamics())
        condensation = Condensation(
            kappa=settings.kappa,
            coord=settings.coord,
            adaptive=settings.adaptive,
            rtol_x=settings.rtol_x,
            rtol_thd=settings.rtol_thd
        )
        builder.add_dynamic(condensation)

        products = [
            ParticlesWetSizeSpectrum(v_bins=phys.volume(settings.r_bins_edges)),
            CondensationTimestep(),
            RipeningRate()
        ]

        attributes = environment.init_attributes(
            n_in_dv=settings.n,
            kappa=settings.kappa,
            r_dry=settings.r_dry
        )

        self.core = builder.build(attributes, products)

        self.n_steps = settings.n_steps

    # TODO: make it common with Arabas_and_Shima_2017
    def save(self, output):
        cell_id = 0
        output["r_bins_values"].append(self.core.products["Particles Wet Size Spectrum"].get())
        volume = self.core.particles['volume'].to_ndarray()
        output["r"].append(phys.radius(volume=volume))
        output["S"].append(self.core.environment["RH"][cell_id] - 1)
        output["qv"].append(self.core.environment["qv"][cell_id])
        output["T"].append(self.core.environment["T"][cell_id])
        output["z"].append(self.core.environment["z"][cell_id])
        output["t"].append(self.core.environment["t"][cell_id])
        output["dt_cond_max"].append(self.core.products["dt_cond"].get_max().copy())
        output["dt_cond_min"].append(self.core.products["dt_cond"].get_min().copy())
        self.core.products["dt_cond"].reset()
        output['ripening_rate'].append(self.core.products['ripening_rate'].get()[cell_id].copy())

    def run(self):
        output = {"r": [], "S": [], "z": [], "t": [], "qv": [], "T": [],
                  "r_bins_values": [], "dt_cond_max": [], "dt_cond_min": [], "ripening_rate": []}

        self.save(output)
        for step in range(self.n_steps):
            self.core.run(self.n_substeps)
            self.save(output)
        return output
