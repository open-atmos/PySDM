import jax
from PySDM.physics import si
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.spectra.exponential import Exponential
import numpy as np
from PySDM import Builder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.backends import JAX, CPU, GPU
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum
import time


def test_run_sim():
    with jax.default_device(jax.devices("cpu")[0]):
        n_sd = 2**15
        initial_spectrum = Exponential(norm_factor=8.39e12, scale=1.19e5 * si.um**3)
        attributes = {}
        attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
            initial_spectrum
        ).sample_deterministic(n_sd)

        radius_bins_edges = np.logspace(
            np.log10(10 * si.um), np.log10(5e3 * si.um), num=32
        )

        env = Box(dt=1 * si.s, dv=1e6 * si.m**3)
        builder = Builder(n_sd=n_sd, backend=JAX(), environment=env)
        builder.add_dynamic(
            Coalescence(
                collision_kernel=Golovin(b=1.5e3 / si.s),
                adaptive=False,
                croupier="global",
            )
        )
        products = [
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                radius_bins_edges=radius_bins_edges, name="dv/dlnr"
            )
        ]
        particulator = builder.build(attributes, products)

        # for step in [0]: #, 1200, 2400, 3600]:
        for step in [i for i in range(1, 10)]:  # , 2400, 3600]:
            t0 = time.time()
            particulator.run(step - particulator.n_steps)
            print(time.time() - t0)
