from collections import namedtuple

import numpy as np

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import (
    ConstantMultiplicity,
    Logarithmic,
)
from PySDM.physics import si
from PySDM.products.collision.collision_rates import (
    CoalescenceRatePerGridbox,
    CollisionRateDeficitPerGridbox,
    CollisionRatePerGridbox,
)
from PySDM.products.size_spectral import (
    ParticleVolumeVersusRadiusLogarithmSpectrum,
    VolumeFirstMoment,
    VolumeSecondMoment,
    ZerothMoment,
)


def run_box(settings, steps=None, backend_class=CPU):
    builder = Builder(n_sd=settings.n_sd, backend=backend_class(settings.formulae))
    env = Box(dv=settings.dv, dt=settings.dt)
    builder.set_environment(env)
    env["rhod"] = 1.0
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = Logarithmic(
        settings.spectrum
    ).sample(settings.n_sd)
    coal = Coalescence(
        collision_kernel=settings.kernel,
        coalescence_efficiency=settings.coal_eff,
        adaptive=settings.adaptive,
    )
    builder.add_dynamic(coal)
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
        ),
        ZerothMoment(name="M0"),
        VolumeFirstMoment(name="M1"),
        VolumeSecondMoment(name="M2"),
        CollisionRatePerGridbox(name="cr"),
        CollisionRateDeficitPerGridbox(name="crd"),
        CoalescenceRatePerGridbox(name="cor"),
    )
    core = builder.build(attributes, products)

    if steps is None:
        steps = settings.output_steps
    y = np.ndarray((len(steps), len(settings.radius_bins_edges) - 1))
    mom = np.ndarray((len(steps), 3))
    rates = np.zeros((len(steps), 4))
    # run
    for i, step in enumerate(steps):
        core.run(step - core.n_steps)
        y[i] = core.products["dv/dlnr"].get()[0]
        rates[i, 0] = core.products["cr"].get()
        rates[i, 1] = core.products["crd"].get()
        rates[i, 2] = core.products["cor"].get()
        mom[i, 0] = core.products["M0"].get()
        mom[i, 1] = core.products["M1"].get()
        mom[i, 2] = core.products["M2"].get()

    x = (settings.radius_bins_edges[:-1] / si.micrometres,)[0]

    return namedtuple("_", ("x", "y", "rates", "moments"))(
        x=x, y=y, rates=rates, moments=mom
    )
