from collections import namedtuple

import numpy as np

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence, Collision
from PySDM.environments import Box
from PySDM.formulae import Formulae
from PySDM.initialisation.sampling.spectral_sampling import (
    ConstantMultiplicity,
    Logarithmic,
)
from PySDM.physics import si
from PySDM.products.collision.collision_rates import (
    BreakupRatePerGridbox,
    CoalescenceRatePerGridbox,
    CollisionRateDeficitPerGridbox,
    CollisionRatePerGridbox,
)
from PySDM.products.size_spectral import (
    NumberSizeSpectrum,
    ParticleVolumeVersusRadiusLogarithmSpectrum,
)


def run_box_breakup(
    settings, steps=None, backend_class=CPU, sample_in_radius=False, return_nv=False
):
    env = Box(dv=settings.dv, dt=settings.dt)
    builder = Builder(
        n_sd=settings.n_sd, backend=backend_class(settings.formulae), environment=env
    )
    env["rhod"] = 1.0
    attributes = {}
    if sample_in_radius:
        diams, attributes["multiplicity"] = Logarithmic(settings.spectrum).sample(
            settings.n_sd
        )
        radii = diams / 2
        attributes["volume"] = Formulae().trivia.volume(radius=radii)
    else:
        attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
            settings.spectrum
        ).sample(settings.n_sd)
    breakup = Collision(
        collision_kernel=settings.kernel,
        coalescence_efficiency=settings.coal_eff,
        breakup_efficiency=settings.break_eff,
        fragmentation_function=settings.fragmentation,
        adaptive=settings.adaptive,
        warn_overflows=settings.warn_overflows,
    )
    builder.add_dynamic(breakup)
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
        ),
        NumberSizeSpectrum(radius_bins_edges=settings.radius_bins_edges, name="N(v)"),
        CollisionRatePerGridbox(name="cr"),
        CollisionRateDeficitPerGridbox(name="crd"),
        CoalescenceRatePerGridbox(name="cor"),
        BreakupRatePerGridbox(name="br"),
    )
    core = builder.build(attributes, products)

    if steps is None:
        steps = settings.output_steps
    y = np.ndarray((len(steps), len(settings.radius_bins_edges) - 1))
    if return_nv:
        y2 = np.ndarray((len(steps), len(settings.radius_bins_edges) - 1))
    else:
        y2 = None

    rates = np.zeros((len(steps), 4))
    # run
    for i, step in enumerate(steps):
        core.run(step - core.n_steps)
        y[i] = core.products["dv/dlnr"].get()[0]
        if return_nv:
            y2[i] = core.products["N(v)"].get()[0]
        rates[i, 0] = core.products["cr"].get()
        rates[i, 1] = core.products["crd"].get()
        rates[i, 2] = core.products["cor"].get()
        rates[i, 3] = core.products["br"].get()

    x = (settings.radius_bins_edges[:-1] / si.micrometres,)[0]

    return namedtuple("_", ("x", "y", "y2", "rates"))(x=x, y=y, y2=y2, rates=rates)


def run_box_NObreakup(settings, steps=None, backend_class=CPU):
    env = Box(dv=settings.dv, dt=settings.dt)
    builder = Builder(
        n_sd=settings.n_sd, backend=backend_class(settings.formulae), environment=env
    )
    env["rhod"] = 1.0
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
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
        CollisionRatePerGridbox(name="cr"),
        CollisionRateDeficitPerGridbox(name="crd"),
        CoalescenceRatePerGridbox(name="cor"),
    )
    core = builder.build(attributes, products)

    if steps is None:
        steps = settings.output_steps
    y = np.ndarray((len(steps), len(settings.radius_bins_edges) - 1))
    rates = np.zeros((len(steps), 4))
    # run
    for i, step in enumerate(steps):
        core.run(step - core.n_steps)
        y[i] = core.products["dv/dlnr"].get()[0]
        rates[i, 0] = core.products["cr"].get()
        rates[i, 1] = core.products["crd"].get()
        rates[i, 2] = core.products["cor"].get()

    x = (settings.radius_bins_edges[:-1] / si.micrometres,)[0]

    return (x, y, rates)
