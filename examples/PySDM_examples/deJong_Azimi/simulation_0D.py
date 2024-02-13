from collections import namedtuple

import numpy as np

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products.size_spectral import (
    ParticleVolumeVersusRadiusLogarithmSpectrum,
    VolumeFirstMoment,
    VolumeSecondMoment,
    ZerothMoment,
)


def run_box(settings, backend_class=CPU):
    env = Box(dv=settings.dv, dt=settings.dt)
    builder = Builder(
        n_sd=settings.n_sd, backend=backend_class(settings.formulae), environment=env
    )
    env["rhod"] = settings.rhod
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        settings.spectrum
    ).sample(settings.n_sd)
    builder.add_dynamic(
        Coalescence(
            collision_kernel=settings.kernel,
            coalescence_efficiency=settings.coal_eff,
            adaptive=settings.adaptive,
        )
    )
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
        ),
        ZerothMoment(name="M0"),
        VolumeFirstMoment(name="M1"),
        VolumeSecondMoment(name="M2"),
    )
    particulator = builder.build(attributes, products)

    y = np.ndarray((len(settings.steps), len(settings.radius_bins_edges) - 1))
    mom = np.ndarray((len(settings.steps), 3))
    for i, step in enumerate(settings.steps):
        particulator.run(step - particulator.n_steps)
        y[i] = particulator.products["dv/dlnr"].get()[0]
        mom[i, 0] = particulator.products["M0"].get()
        mom[i, 1] = particulator.products["M1"].get()
        mom[i, 2] = particulator.products["M2"].get()

    return namedtuple("_", ("radius_bins_left_edges", "dv_dlnr", "moments"))(
        radius_bins_left_edges=settings.radius_bins_edges[:-1], dv_dlnr=y, moments=mom
    )
