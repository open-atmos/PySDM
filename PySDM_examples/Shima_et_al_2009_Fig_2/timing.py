"""
Created at 08.08.2019
"""

from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.spectral_sampling import ConstantMultiplicity
from PySDM_examples.Shima_et_al_2009_Fig_2.settings import Settings


def run(settings):
    builder = Builder(n_sd=settings.n_sd, backend=settings.backend)
    builder.set_environment(Box(dv=settings.dv, dt=settings.dt))
    attributes = {}
    attributes['volume'], attributes['n'] = ConstantMultiplicity(settings.spectrum).sample(settings.n_sd)
    builder.add_dynamic(Coalescence(settings.kernel))
    particles = builder.build(attributes)

    states = {}
    for step in settings.steps:
        particles.run(step - particles.n_steps)

    return states, particles.stats


# TIP: try with: python -O
from PySDM.backends.numba.numba import Numba
from PySDM.backends.thrustRTC.thrustRTC import ThrustRTC

settings = Settings()
settings.steps = [100, 3600]

times = {}
for backend in (ThrustRTC, Numba):
    settings.backend = backend
    nsds = [2 ** n for n in range(12, 19, 3)]
    key = backend.__name__
    times[key] = []
    for sd in nsds:
        settings.n_sd = sd
        _, stats = run(settings)
        times[key].append(stats.wall_times[-1])

from matplotlib import pyplot as plt
for backend, t in times.items():
    plt.plot(nsds, t, label=backend)
plt.legend()
plt.loglog()
plt.show()
