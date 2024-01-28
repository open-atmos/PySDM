import numpy as np
from PySDM_examples.Srivastava_1982.settings import SimProducts

from PySDM import Builder, Formulae
from PySDM.environments import Box
from PySDM.products import SuperDropletCountPerGridbox, VolumeFirstMoment, ZerothMoment


class Simulation:
    def __init__(
        self, n_steps, settings, collision_dynamic=None, double_precision=True
    ):
        self.collision_dynamic = collision_dynamic
        self.settings = settings
        self.n_steps = n_steps

        self.double_precision = double_precision

        self.simulation_res = {
            n_sd: {prod: {} for prod in self.settings.prods}
            for n_sd in self.settings.n_sds
        }

    def build(self, n_sd, seed, products):
        env = Box(dt=self.settings.dt, dv=self.settings.dv)
        builder = Builder(
            backend=self.settings.backend_class(
                formulae=Formulae(
                    constants={"rho_w": self.settings.rho},
                    fragmentation_function="ConstantMass",
                    seed=seed,
                ),
                double_precision=self.double_precision,
            ),
            n_sd=n_sd,
            environment=env,
        )
        builder.add_dynamic(self.collision_dynamic)
        particulator = builder.build(
            products=products,
            attributes={
                "multiplicity": np.full(n_sd, self.settings.total_number_0 / n_sd),
                "volume": np.full(
                    n_sd,
                    self.settings.total_volume / self.settings.total_number_0,
                ),
            },
        )
        return particulator

    def run_convergence_analysis(self, x, seeds):
        for n_sd in self.settings.n_sds:
            for seed in seeds:
                products = (
                    SuperDropletCountPerGridbox(
                        name=SimProducts.PySDM.super_particle_count.name
                    ),
                    VolumeFirstMoment(name=SimProducts.PySDM.total_volume.name),
                    ZerothMoment(name=SimProducts.PySDM.total_numer.name),
                )

                particulator = self.build(n_sd, seed, products)

                for prod in self.settings.prods:
                    self.simulation_res[n_sd][prod][seed] = np.full(
                        self.n_steps + 1, -np.inf
                    )

                for step in range(len(x)):
                    if step != 0:
                        particulator.run(steps=1)
                    for prod in self.settings.prods:
                        self.simulation_res[n_sd][prod][seed][step] = (
                            particulator.products[prod].get()
                        )

                np.testing.assert_allclose(
                    actual=self.simulation_res[n_sd][
                        SimProducts.PySDM.total_volume.name
                    ][seed],
                    desired=self.settings.total_volume,
                    rtol=1e-3,
                )

        return self.simulation_res
